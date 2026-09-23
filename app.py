import gradio_client.utils as _client_utils

# ============================================================
# GRADIO / PYDANTIC WORKAROUND
# ============================================================

_original_json_schema_to_python_type = (
    _client_utils._json_schema_to_python_type
)


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"

    return _original_json_schema_to_python_type(
        schema,
        defs,
    )


_client_utils._json_schema_to_python_type = (
    _safe_json_schema_to_python_type
)


"""
app.py — Watermark Extractor

Public Space with a 10-image-per-session limit.

Important extraction behavior:

- Every image is processed independently.
- Same Side ID is allowed on multiple images.
- Coordinates always belong to the current image.
- Previous extracted results are passed to the extractor only
  to detect suspicious exact coordinate reuse.
- Duplicate Side IDs are NOT rejected.
"""


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv

load_dotenv()

import os
import io
import time
import shutil

import pandas as pd
import gradio as gr

from core.extractor import extract_watermark

from core.distance import (
    load_master_from_excel,
    validate_against_master,
)

from utils.helpers import (
    clean_coord,
    force_string,
    clean_side_id,
    truncate_to_8_decimals,
    _cap_longest_side,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "temp_outputs",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# ABUSE PROTECTION
# ============================================================

MAX_IMAGES_PER_SESSION = 10


# ============================================================
# DESIGN TOKENS
# ============================================================

INK = "#172033"
MUTED = "#64748b"
BORDER = "#e2e8f0"
BLUE = "#2563eb"

SANS = "font-family:'Work Sans',sans-serif;"


STATUS_CFG = {
    "queued": {
        "label": "Queued",
        "fg": "#475569",
        "bg": "#f1f5f9",
    },

    "scanning": {
        "label": "Processing",
        "fg": "#1d4ed8",
        "bg": "#dbeafe",
    },

    "done": {
        "label": "Completed",
        "fg": "#166534",
        "bg": "#dcfce7",
    },

    "failed": {
        "label": "Failed",
        "fg": "#b91c1c",
        "bg": "#fee2e2",
    },
}


VERIFY_STATUS_STYLE = {
    "Within 300m": {
        "fg": "#166534",
        "bg": "#dcfce7",
    },

    "Outside 300m": {
        "fg": "#c2410c",
        "bg": "#ffedd5",
    },

    "Unknown Site": {
        "fg": "#475569",
        "bg": "#f1f5f9",
    },

    "Missing Coordinates": {
        "fg": "#475569",
        "bg": "#f1f5f9",
    },

    "Invalid Coordinate Format": {
        "fg": "#b91c1c",
        "bg": "#fee2e2",
    },
}


# ============================================================
# HTML HELPERS
# ============================================================

def _chip_html(status):

    cfg = STATUS_CFG.get(
        status,
        STATUS_CFG["failed"],
    )

    return (
        f'<span style="{SANS}'
        f'color:{cfg["fg"]};'
        f'background:{cfg["bg"]};'
        f'font-size:12px;'
        f'font-weight:600;'
        f'padding:3px 10px;'
        f'border-radius:999px;'
        f'display:inline-block;">'
        f'{cfg["label"]}'
        f'</span>'
    )


def _cell_html(v):

    if v is None or str(v).strip() == "":
        return (
            f'<span style="{SANS}'
            f'color:#cbd5e1;'
            f'font-size:13px;">'
            f'&mdash;'
            f'</span>'
        )

    return (
        f'<span style="{SANS}'
        f'color:{INK};'
        f'font-size:13px;">'
        f'{v}'
        f'</span>'
    )


# ============================================================
# EXTRACTION STATS
# ============================================================

def _stats_html(entries):

    done = [
        e
        for e in entries
        if e.get("status") == "done"
    ]

    failed = [
        e
        for e in entries
        if e.get("status") == "failed"
    ]

    queued = [
        e
        for e in entries
        if e.get("status")
        in ("queued", "scanning")
    ]

    stats = [
        ("Total", len(entries), False),
        ("Extracted", len(done), True),
        ("In Queue", len(queued), False),
        ("Failed", len(failed), False),
    ]

    cards = ""

    for label, val, accent in stats:

        color = (
            BLUE
            if accent
            else INK
        )

        cards += (
            f'<article style="'
            f'flex:1;'
            f'background:#ffffff;'
            f'border:1px solid {BORDER};'
            f'border-radius:12px;'
            f'padding:16px 20px;">'

            f'<p style="{SANS}'
            f'color:{MUTED};'
            f'font-size:13px;'
            f'margin:0;">'
            f'{label}'
            f'</p>'

            f'<p style="{SANS}'
            f'color:{color};'
            f'font-size:24px;'
            f'font-weight:700;'
            f'margin:6px 0 0 0;">'
            f'{val}'
            f'</p>'

            f'</article>'
        )

    return (
        f'<div style="'
        f'display:flex;'
        f'gap:14px;'
        f'margin-bottom:18px;">'
        f'{cards}'
        f'</div>'
    )


# ============================================================
# EXTRACTION TABLE
# ============================================================

def _table_html(entries):

    if not entries:

        return (
            f'<div style="'
            f'display:flex;'
            f'flex-direction:column;'
            f'align-items:center;'
            f'justify-content:center;'
            f'gap:12px;'
            f'padding:80px 0;'
            f'border:1px solid {BORDER};'
            f'border-radius:12px;'
            f'background:#ffffff;">'

            f'<div style="{SANS}'
            f'font-size:15px;'
            f'font-weight:600;'
            f'color:{INK};">'
            f'No images in queue'
            f'</div>'

            f'<p style="{SANS}'
            f'color:{MUTED};'
            f'font-size:13px;">'
            f'Upload files to begin'
            f'</p>'

            f'</div>'
        )

    headers = [
        "No.",
        "Image",
        "Side ID",
        "Latitude",
        "Longitude",
        "MP No.",
        "Status",
    ]

    head_cells = "".join(
        f'<th style="{SANS}'
        f'font-size:12px;'
        f'color:{MUTED};'
        f'text-transform:uppercase;'
        f'font-weight:600;'
        f'background:#f8fafc;'
        f'padding:12px 20px;'
        f'text-align:left;">'
        f'{h}'
        f'</th>'
        for h in headers
    )

    rows = ""

    for i, e in enumerate(entries):

        bg = (
            "#eff6ff"
            if e.get("status") == "scanning"
            else "transparent"
        )

        err = ""

        if e.get("error"):

            err = (
                f'<div style="{SANS}'
                f'color:#b91c1c;'
                f'font-size:11px;'
                f'margin-top:2px;">'
                f'{str(e.get("error", ""))[:120]}'
                f'</div>'
            )

        rows += (
            f'<tr style="'
            f'border-bottom:1px solid #eef2f7;'
            f'background:{bg};">'

            f'<td style="{SANS}'
            f'color:{MUTED};'
            f'font-size:12px;'
            f'padding:12px 20px;">'
            f'{str(i + 1).zfill(2)}'
            f'</td>'

            f'<td style="{SANS}'
            f'font-size:13px;'
            f'color:{INK};'
            f'padding:12px 20px;'
            f'max-width:200px;'
            f'overflow:hidden;'
            f'text-overflow:ellipsis;'
            f'white-space:nowrap;">'
            f'{e.get("image", "")}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(e.get("side_id"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(e.get("latitude"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(e.get("longitude"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(e.get("mp_number"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_chip_html(e.get("status", "failed"))}'
            f'{err}'
            f'</td>'

            f'</tr>'
        )

    return (
        f'<div style="'
        f'background:#ffffff;'
        f'border:1px solid {BORDER};'
        f'border-radius:12px;'
        f'overflow:hidden;">'

        f'<div style="'
        f'overflow-y:auto;'
        f'max-height:65vh;'
        f'overflow-x:auto;">'

        f'<table style="'
        f'width:100%;'
        f'border-collapse:collapse;'
        f'min-width:700px;">'

        f'<thead>'
        f'<tr>{head_cells}</tr>'
        f'</thead>'

        f'<tbody>'
        f'{rows}'
        f'</tbody>'

        f'</table>'
        f'</div>'
        f'</div>'
    )


# ============================================================
# DISTANCE STATUS CHIP
# ============================================================

def _verify_status_chip(status):

    """
    Multiple images may have the same Side ID.

    Duplicate Side IDs are NOT treated as an error.
    """

    base_status = (
        status or ""
    ).split(" (")[0]

    cfg = VERIFY_STATUS_STYLE.get(
        base_status,
        {
            "fg": "#475569",
            "bg": "#f1f5f9",
        },
    )

    return (
        f'<span style="{SANS}'
        f'color:{cfg["fg"]};'
        f'background:{cfg["bg"]};'
        f'font-size:12px;'
        f'font-weight:600;'
        f'padding:3px 10px;'
        f'border-radius:999px;'
        f'display:inline-block;">'
        f'{base_status}'
        f'</span>'
    )


# ============================================================
# DISTANCE VERIFICATION TABLE
# ============================================================

def _verification_table_html(results):

    if not results:

        return (
            f'<div style="{SANS}'
            f'padding:40px;'
            f'text-align:center;'
            f'color:{MUTED};'
            f'border:1px solid {BORDER};'
            f'border-radius:12px;'
            f'background:#ffffff;">'
            f'No results yet.'
            f'</div>'
        )

    headers = [
        "Image",
        "Side ID",
        "Latitude",
        "Longitude",
        "Master Lat",
        "Master Lon",
        "Distance (m)",
        "Status",
    ]

    head_cells = "".join(
        f'<th style="{SANS}'
        f'font-size:12px;'
        f'color:{MUTED};'
        f'text-transform:uppercase;'
        f'font-weight:600;'
        f'background:#f8fafc;'
        f'padding:12px 20px;'
        f'text-align:left;">'
        f'{h}'
        f'</th>'
        for h in headers
    )

    rows = ""

    for r in results:

        rows += (
            f'<tr style="'
            f'border-bottom:1px solid #eef2f7;">'

            f'<td style="{SANS}'
            f'font-size:13px;'
            f'color:{INK};'
            f'padding:12px 20px;'
            f'max-width:200px;'
            f'overflow:hidden;'
            f'text-overflow:ellipsis;'
            f'white-space:nowrap;">'
            f'{r.get("image", "")}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(r.get("side_id"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(r.get("latitude"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(r.get("longitude"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(r.get("master_latitude"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(r.get("master_longitude"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_cell_html(r.get("distance_m"))}'
            f'</td>'

            f'<td style="padding:12px 20px;">'
            f'{_verify_status_chip(r.get("status", ""))}'
            f'</td>'

            f'</tr>'
        )

    return (
        f'<div style="'
        f'background:#ffffff;'
        f'border:1px solid {BORDER};'
        f'border-radius:12px;'
        f'overflow:hidden;'
        f'margin-top:14px;">'

        f'<div style="'
        f'overflow-y:auto;'
        f'max-height:65vh;'
        f'overflow-x:auto;">'

        f'<table style="'
        f'width:100%;'
        f'border-collapse:collapse;'
        f'min-width:820px;">'

        f'<thead>'
        f'<tr>{head_cells}</tr>'
        f'</thead>'

        f'<tbody>'
        f'{rows}'
        f'</tbody>'

        f'</table>'
        f'</div>'
        f'</div>'
    )


# ============================================================
# CORE EXTRACTION WRAPPER
# ============================================================

def _extract_one(
    image_bytes,
    filename,
    master_ids=None,
    previous_results=None,
):
    """
    Extract ONE image.

    previous_results is intentionally passed to the extractor.

    This lets the extractor detect:

        Image A:
        1683 -> coordinate X

        Image B:
        1683 -> coordinate X

    and perform a second independent verification.

    IMPORTANT:

    Same Side ID alone is NOT considered an error.
    """

    return extract_watermark(
        image_bytes,
        filename,
        master_ids=master_ids,
        previous_results=previous_results,
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_images(files):
    """
    Gradio handler.

    Every image is processed independently.

    previous_results keeps accepted observations from earlier
    images in this batch.
    """

    if not files:

        yield (
            None,
            _stats_html([]),
            _table_html([]),
            [],
        )

        return

    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    if len(files) > MAX_IMAGES_PER_SESSION:

        gr.Warning(
            f"Maximum {MAX_IMAGES_PER_SESSION} images "
            f"per session. "
            f"Only the first "
            f"{MAX_IMAGES_PER_SESSION} "
            f"will be processed."
        )

        files = files[
            :MAX_IMAGES_PER_SESSION
        ]

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    entries = [
        {
            "image": os.path.basename(f.name),
            "path": f.name,
            "status": "queued",
        }
        for f in files
    ]

    # IMPORTANT:
    #
    # This list contains the extraction result of previous
    # images in THIS batch.
    #
    # It does NOT mean duplicate Side IDs are forbidden.
    #
    # It is only used to detect suspicious coordinate reuse.

    previous_results = []

    yield (
        None,
        _stats_html(entries),
        _table_html(entries),
        entries,
    )

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for i, entry in enumerate(entries):

        entries[i]["status"] = "scanning"

        yield (
            None,
            _stats_html(entries),
            _table_html(entries),
            entries,
        )

        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            with open(
                entry["path"],
                "rb",
            ) as fh:

                image_bytes = fh.read()

            # ------------------------------------------------
            # EXTRACT CURRENT IMAGE
            # ------------------------------------------------

            data = _extract_one(
                image_bytes,
                entry["image"],
                master_ids=None,

                # THIS IS THE IMPORTANT CHANGE
                previous_results=previous_results,
            )

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            if isinstance(data, dict):

                entries[i].update(data)

            # ------------------------------------------------
            # DETERMINE SUCCESS
            # ------------------------------------------------

            got_required_data = (
                data.get("side_id")
                not in (
                    None,
                    "",
                    "Too blurry to extract",
                )
                and
                data.get("latitude")
                not in (
                    None,
                    "",
                )
                and
                data.get("longitude")
                not in (
                    None,
                    "",
                )
            )

            if got_required_data:

                entries[i]["status"] = "done"

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Add this image's extraction to previous_results
                # ONLY AFTER it has successfully completed.
                #
                # This allows the NEXT image to compare against it.
                # ------------------------------------------------

                previous_results.append(
                    {
                        "image": entry["image"],
                        "side_id": data.get("side_id"),
                        "latitude": data.get("latitude"),
                        "longitude": data.get("longitude"),
                        "mp_number": data.get("mp_number"),
                    }
                )

            else:

                entries[i]["status"] = "failed"

                entries[i]["error"] = (
                    "Could not extract valid "
                    "Side ID + latitude + longitude"
                )

        except Exception as e:

            entries[i]["status"] = "failed"

            entries[i]["error"] = str(e)[:120]

        # ----------------------------------------------------
        # UPDATE UI
        # ----------------------------------------------------

        yield (
            None,
            _stats_html(entries),
            _table_html(entries),
            entries,
        )

        if i < len(entries) - 1:

            time.sleep(1)

    # ========================================================
    # BUILD EXCEL
    # ========================================================

    df = pd.DataFrame(
        [
            {
                k: e.get(k)
                for k in [
                    "image",
                    "side_id",
                    "latitude",
                    "longitude",
                    "mp_number",
                ]
            }
            for e in entries
        ]
    )

    excel_buffer = io.BytesIO()

    df.to_excel(
        excel_buffer,
        index=False,
    )

    excel_buffer.seek(0)

    output_path = os.path.join(
        OUTPUT_DIR,
        "extracted_data.xlsx",
    )

    with open(
        output_path,
        "wb",
    ) as f:

        f.write(
            excel_buffer.getvalue()
        )

    yield (
        output_path,
        _stats_html(entries),
        _table_html(entries),
        entries,
    )


# ============================================================
# DISTANCE VALIDATION
# ============================================================

def check_distance_click(
    master_file,
    entries,
):
    """
    Validate EVERY extracted image independently.

    IMPORTANT:

    If these exist:

        1683 image A
        1683 image B
        1683 image C

    all three are kept.

    Their coordinates are NOT merged.

    validate_against_master() receives all entries separately.
    """

    empty_box = (
        f'<div style="{SANS}'
        f'padding:40px;'
        f'text-align:center;'
        f'color:{MUTED};'
        f'border:1px solid {BORDER};'
        f'border-radius:12px;'
        f'background:#ffffff;">'
        f'{{msg}}'
        f'</div>'
    )

    if not entries:

        return (
            None,
            empty_box.format(
                msg=(
                    'No processed images yet — '
                    'run "Extract Data" first.'
                )
            ),
        )

    if master_file is None:

        return (
            None,
            empty_box.format(
                msg=(
                    "Upload your master site file first."
                )
            ),
        )

    try:

        # ----------------------------------------------------
        # PRESERVE ORIGINAL MASTER FILE EXTENSION
        # ----------------------------------------------------

        original_name = os.path.basename(
            master_file.name
        )

        stable_path = os.path.join(
            OUTPUT_DIR,
            original_name,
        )

        shutil.copy(
            master_file.name,
            stable_path,
        )

        # ----------------------------------------------------
        # LOAD MASTER
        # ----------------------------------------------------

        master_df = load_master_from_excel(
            stable_path
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT deduplicate entries.
        #
        # Every image is an independent observation.
        # ----------------------------------------------------

        results = validate_against_master(
            entries,
            master_df,
        )

    except Exception as e:

        return (
            None,
            empty_box.format(
                msg=(
                    f"Could not process master file: {e}"
                )
            ),
        )

    # --------------------------------------------------------
    # BUILD VALIDATION EXCEL
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "validated_output.xlsx",
    )

    df.to_excel(
        output_path,
        index=False,
    )

    return (
        output_path,
        _verification_table_html(results),
    )


# ============================================================
# CLEAR ALL
# ============================================================

def clear_all():

    """
    Reset the complete application state.
    """

    return (
        None,                         # image input
        None,                         # extracted Excel
        _stats_html([]),              # stats
        _table_html([]),              # extraction table
        [],                           # results_state
        None,                         # master file
        None,                         # validated Excel
        "",                           # verification table
    )


# ============================================================
# CUSTOM CSS
# ============================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700&display=swap');

* {
    box-sizing: border-box;
}

body,
.gradio-container {
    background: #f7f9fc !important;
    font-family: 'Work Sans', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

.sidebar {
    background: #ffffff !important;
    padding: 28px 24px !important;
    border-right: 1px solid #e2e8f0 !important;
    min-height: 100vh;
}

.main-panel {
    background: transparent !important;
    padding: 24px 28px !important;
}

.upload-zone {
    border: 1.5px dashed #cbd5e1 !important;
    background: #f8fafc !important;
    border-radius: 12px !important;
}

.process-btn,
.process-btn button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Work Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
}

.process-btn:hover button,
.process-btn button:hover {
    background: #1d4ed8 !important;
}

.clear-btn,
.clear-btn button {
    background: none !important;
    border: none !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
}

.download-area {
    background: #ecfdf5 !important;
    border: 1px solid #6ee7b7 !important;
    border-radius: 8px !important;
}

.download-area * {
    color: #047857 !important;
}
"""


# ============================================================
# GRADIO LAYOUT
# ============================================================

with gr.Blocks(
    css=css,
    title="Watermark Extractor",
) as app:

    with gr.Row():

        # ====================================================
        # SIDEBAR
        # ====================================================

        with gr.Column(
            scale=0,
            min_width=280,
            elem_classes=["sidebar"],
        ):

            gr.HTML(
                f"""
                <div style="
                    padding-bottom:24px;
                    border-bottom:1px solid #e2e8f0;
                    display:flex;
                    align-items:center;
                    gap:12px;
                ">

                    <div style="
                        width:36px;
                        height:36px;
                        border-radius:10px;
                        background:{BLUE};
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        color:#ffffff;
                        font-size:18px;
                    ">
                        📡
                    </div>

                    <div>

                        <h1 style="
                            {SANS}
                            color:{INK};
                            font-size:16px;
                            font-weight:700;
                            line-height:1.2;
                            margin:0;
                        ">
                            Watermark Extractor
                        </h1>

                        <p style="
                            {SANS}
                            color:{MUTED};
                            font-size:12px;
                            margin-top:2px;
                        ">
                            Field survey tool
                        </p>

                    </div>

                </div>

                <p style="
                    {SANS}
                    color:{MUTED};
                    font-size:13px;
                    margin-top:16px;
                    line-height:1.5;
                ">
                    Upload survey photos to extract site ID,
                    coordinates, and MP number.

                    Limit:
                    {MAX_IMAGES_PER_SESSION}
                    images per session.
                </p>
                """
            )

            image_input = gr.File(
                label="Drop images · JPG · JPEG · PNG",
                file_types=[
                    ".jpg",
                    ".jpeg",
                    ".png",
                ],
                file_count="multiple",
                elem_classes=["upload-zone"],
            )

            process_btn = gr.Button(
                "Extract Data",
                elem_classes=["process-btn"],
            )

            excel_output = gr.File(
                label="Download Excel",
                elem_classes=["download-area"],
            )

            clear_btn = gr.Button(
                "Clear all",
                elem_classes=["clear-btn"],
                size="sm",
            )

        # ====================================================
        # MAIN PANEL
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes=["main-panel"],
        ):

            gr.HTML(
                f"""
                <div style="margin-bottom:18px;">

                    <h2 style="
                        {SANS}
                        color:{INK};
                        font-size:20px;
                        font-weight:700;
                        margin:0;
                    ">
                        Image Processing
                    </h2>

                    <p style="
                        {SANS}
                        color:{MUTED};
                        font-size:13px;
                        margin-top:4px;
                    ">
                        Upload images and extract structured
                        information automatically.
                    </p>

                </div>
                """
            )

            stats_html = gr.HTML(
                value=_stats_html([])
            )

            table_html = gr.HTML(
                value=_table_html([])
            )

            # IMPORTANT:
            # Stores extraction records for distance validation.
            results_state = gr.State([])

            # =================================================
            # DISTANCE VERIFICATION
            # =================================================

            gr.HTML(
                f"""
                <div style="
                    margin-top:28px;
                    margin-bottom:14px;
                ">

                    <h2 style="
                        {SANS}
                        color:{INK};
                        font-size:20px;
                        font-weight:700;
                        margin:0;
                    ">
                        Distance Verification
                    </h2>

                    <p style="
                        {SANS}
                        color:{MUTED};
                        font-size:13px;
                        margin-top:4px;
                    ">
                        Upload your master site file to check
                        whether each extracted location is within
                        300m of the recorded site.
                    </p>

                </div>
                """
            )

            with gr.Row():

                master_file_input = gr.File(
                    label="Master site file (.xlsx / .csv)",
                    file_types=[
                        ".xlsx",
                        ".csv",
                    ],
                    file_count="single",
                )

                check_distance_btn = gr.Button(
                    "Check Distance",
                    elem_classes=["process-btn"],
                )

            gr.HTML(
                f"""
                <p style="
                    {SANS}
                    color:{MUTED};
                    font-size:12px;
                    margin-top:-4px;
                ">
                    Each image is checked separately.

                    Multiple images may have the same Side ID;
                    their own extracted coordinates are used to
                    calculate the distance for each image.
                </p>
                """
            )

            validated_output_file = gr.File(
                label="Download Validated Report"
            )

            verification_table = gr.HTML(
                value=""
            )

    # ========================================================
    # EVENTS
    # ========================================================

    process_btn.click(
        fn=process_images,
        inputs=[
            image_input,
        ],
        outputs=[
            excel_output,
            stats_html,
            table_html,
            results_state,
        ],
    )

    # ========================================================
    # CLEAR
    # ========================================================

    clear_btn.click(
        fn=clear_all,
        outputs=[
            image_input,
            excel_output,
            stats_html,
            table_html,
            results_state,
            master_file_input,
            validated_output_file,
            verification_table,
        ],
    )

    # ========================================================
    # DISTANCE
    # ========================================================

    check_distance_btn.click(
        fn=check_distance_click,
        inputs=[
            master_file_input,
            results_state,
        ],
        outputs=[
            validated_output_file,
            verification_table,
        ],
    )


# ============================================================
# START
# ============================================================
import os

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )