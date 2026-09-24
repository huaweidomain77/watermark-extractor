"""
core/batch.py

Batch runner for extract_watermark().

- 1 image  -> plain sequential call, no thread pool overhead.
- N images -> one thread per image (up to MAX_WORKERS at a time).

Why threads and not processes:
    Each call to extract_watermark() spends almost all of its time
    waiting on network I/O (the OpenAI API), not doing local CPU
    work. Threads release the GIL while waiting on I/O, so this is
    exactly the situation threads are good at -- no multiprocessing
    needed.

Thread-safety notes:
    extractor.py's `previous_results` argument is used to detect a
    Side ID + exact coordinates repeated across images in the same
    batch. It's a shared, growing list, so under concurrency it needs
    a lock:
      - each worker takes a SNAPSHOT of previous_results (under lock)
        before its own extraction starts, so its duplicate-check sees
        a consistent list even while other threads are still running.
      - each worker appends its own result (under lock) once done.
    This means the duplicate check is "best effort" under threading
    (a result that finishes microseconds after another might not see
    it yet) rather than a strict sequential guarantee -- that's an
    acceptable trade-off for catching accidental copy/paste-style
    duplicates, which is what it's for.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.extractor import extract_watermark


# Cap concurrent OpenAI calls. Raise this only after checking your
# account's actual rate limit -- too high and every thread starts
# hitting 429s together instead of helping.
MAX_WORKERS = 5


def process_batch(images, master_ids=None, max_workers=MAX_WORKERS,
                  on_result=None):
    """
    Extract watermark data from a list of images.

    Parameters
    ----------
    images:
        List of (image_bytes, filename) tuples, in the order you
        want them to appear in the output.
    master_ids:
        Optional set/list of known Side IDs, passed straight through
        to extract_watermark().
    max_workers:
        Max number of images processed at once when there's more
        than one image. Ignored for a single image.
    on_result:
        Optional callback: on_result(index, result). Called as each
        image finishes, useful for updating a progress UI (e.g. your
        Gradio frontend) as results arrive rather than waiting for
        the whole batch.

    Returns
    -------
    List of result dicts, in the SAME ORDER as `images` (not
    completion order), each with an "image" key set to its filename.
    """

    if not images:
        return []

    # ---- Single image: no threading needed ----
    if len(images) == 1:
        image_bytes, filename = images[0]
        result = extract_watermark(
            image_bytes, filename,
            master_ids=master_ids,
            previous_results=[],
        )
        result["image"] = filename
        if on_result:
            on_result(0, result)
        return [result]

    # ---- Multiple images: one thread per image, capped ----
    results = [None] * len(images)
    previous_results = []
    lock = threading.Lock()

    def _worker(index, image_bytes, filename):
        with lock:
            snapshot = list(previous_results)

        result = extract_watermark(
            image_bytes, filename,
            master_ids=master_ids,
            previous_results=snapshot,
        )
        result["image"] = filename

        with lock:
            previous_results.append(result)

        return index, result

    workers = min(max_workers, len(images))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_worker, i, image_bytes, filename)
            for i, (image_bytes, filename) in enumerate(images)
        ]

        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            if on_result:
                on_result(index, result)

    return results
