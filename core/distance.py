"""
core/distance.py

Haversine distance + validation against the master site file.

Matching rules:
    1. Match ONLY against the master's Site ID column.
    2. Match Site IDs exactly.
    3. Do NOT search the whole spreadsheet for an ID.
    4. Preserve prefixes such as S-, N-, CII-, etc.
    5. Multiple uploaded images with the same Side ID are allowed.
    6. If the master contains multiple rows for the same Site ID,
       calculate distance to every matching master row and use
       the nearest one.
"""

import math
import pandas as pd


MAX_M = 300  # meters


# ============================================================
# HAVERSINE
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Return great-circle distance in METERS."""

    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return R * c


# ============================================================
# SITE ID NORMALIZATION
# ============================================================

def _normalize_site_id(value):
    """
    Normalize a Site ID without changing its identity.

    IMPORTANT:
        Prefixes are preserved.

    Examples:
        ' S-7614 '  -> 'S-7614'
        's-7614'    -> 'S-7614'
        'N-4299'    -> 'N-4299'
        'CII-2842'  -> 'CII-2842'
        '1683'      -> '1683'
        1683        -> '1683'

    We intentionally DO NOT do:

        S-7614 -> 7614
        N-4299 -> 4299

    because these may represent different Site IDs.
    """

    if value is None:
        return ""

    # Handle pandas NaN / NA
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    s = str(value).strip().upper()

    # Remove accidental trailing .0 from numeric Excel values.
    #
    # Example:
    #     1683.0 -> 1683
    #
    # But do NOT modify IDs such as:
    #     S-7614
    #     CII-2842
    if s.endswith(".0"):
        numeric_part = s[:-2]

        if numeric_part.isdigit():
            s = numeric_part

    return s


# ============================================================
# MASTER FILE LOADING
# ============================================================

def load_master_from_excel(master_path):
    """
    Load the master site file from CSV or Excel.

    The current master file contains:

        2G Site ID
        Region
        Latitude
        Longitude
        Final DG
        Guarded/Unguarded

    Only these three columns are required:

        2G Site ID
        Latitude
        Longitude

    Returns a DataFrame with:

        master_site_id
        master_latitude
        master_longitude
    """

    # --------------------------------------------------------
    # Read CSV first.
    #
    # dtype=str is intentional.
    #
    # It prevents pandas from converting Site IDs into numbers
    # before we normalize them.
    # --------------------------------------------------------

    try:
        raw = pd.read_csv(
            master_path,
            dtype=str,
            keep_default_na=False
        )

    except Exception:
        try:
            raw = pd.read_excel(
                master_path,
                dtype=str
            )

        except Exception as e:
            raise ValueError(
                "Could not read master file as CSV or Excel: "
                f"{e}"
            )

    # --------------------------------------------------------
    # Normalize column names ONLY for identifying columns.
    # --------------------------------------------------------

    def _norm_column(value):
        return (
            str(value)
            .strip()
            .lower()
            .replace(" ", "_")
        )

    normalized = {
        original: _norm_column(original)
        for original in raw.columns
    }

    site_col = None
    lat_col = None
    lon_col = None

    # --------------------------------------------------------
    # Find Site ID column.
    #
    # Your actual master uses:
    #     2G Site ID
    # --------------------------------------------------------

    for original, norm in normalized.items():

        if site_col is None and norm in (
            "2g_site_id",
            "site_id",
            "siteid",
            "site",
            "id",
        ):
            site_col = original

        if lat_col is None and norm in (
            "latitude",
            "lat",
            "ref_lat",
            "y",
        ):
            lat_col = original

        if lon_col is None and norm in (
            "longitude",
            "lon",
            "lng",
            "ref_lng",
            "x",
        ):
            lon_col = original

    # --------------------------------------------------------
    # Validate required columns.
    # --------------------------------------------------------

    if site_col is None or lat_col is None or lon_col is None:

        raise ValueError(
            "Could not find Site ID / Latitude / Longitude "
            f"columns in {master_path}.\n\n"
            f"Found columns: {list(raw.columns)}\n\n"
            f"Detected:\n"
            f"  Site ID = {site_col!r}\n"
            f"  Latitude = {lat_col!r}\n"
            f"  Longitude = {lon_col!r}"
        )

    # --------------------------------------------------------
    # Keep ONLY the columns required for distance matching.
    #
    # This is important:
    #
    # The distance system will NEVER search other columns such
    # as Region, Final DG, etc. for a Site ID.
    # --------------------------------------------------------

    master_df = raw.rename(
        columns={
            site_col: "master_site_id",
            lat_col: "master_latitude",
            lon_col: "master_longitude",
        }
    )[
        [
            "master_site_id",
            "master_latitude",
            "master_longitude",
        ]
    ].copy()

    # --------------------------------------------------------
    # Normalize Site IDs.
    # --------------------------------------------------------

    master_df["master_site_id"] = (
        master_df["master_site_id"]
        .apply(_normalize_site_id)
    )

    # --------------------------------------------------------
    # Convert coordinates to numbers.
    # --------------------------------------------------------

    master_df["master_latitude"] = pd.to_numeric(
        master_df["master_latitude"],
        errors="coerce"
    )

    master_df["master_longitude"] = pd.to_numeric(
        master_df["master_longitude"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove unusable rows.
    # --------------------------------------------------------

    master_df = master_df[
        (master_df["master_site_id"] != "")
        & master_df["master_latitude"].notna()
        & master_df["master_longitude"].notna()
    ].copy()

    # Reset index for clean iteration later.
    master_df.reset_index(drop=True, inplace=True)

    return master_df


# ============================================================
# DISTANCE VALIDATION
# ============================================================

def validate_against_master(entries, master_df):
    """
    Validate every extracted image independently.

    IMPORTANT:

    If entries contain:

        image A -> Side ID 1683
        image B -> Side ID 1683

    BOTH entries are processed.

    We do NOT deduplicate uploaded images by Side ID.

    If the master itself contains multiple rows for the same
    exact Site ID, distance is calculated against every matching
    master row and the nearest master coordinate is selected.

    Returns a list of result dictionaries.
    """

    results = []

    # --------------------------------------------------------
    # Find duplicate IDs INSIDE THE MASTER ONLY.
    #
    # This has nothing to do with repeated uploaded images.
    # --------------------------------------------------------

    master_duplicate_ids = set(
        master_df.loc[
            master_df["master_site_id"].duplicated(keep=False),
            "master_site_id",
        ].unique()
    )

    # --------------------------------------------------------
    # Process EVERY uploaded entry independently.
    # --------------------------------------------------------

    for e in entries:

        site_id = _normalize_site_id(
            e.get("side_id")
        )

        cap_lat = e.get("latitude")
        cap_lon = e.get("longitude")

        base = {
            "image": e.get("image"),
            "side_id": site_id,
            "latitude": cap_lat,
            "longitude": cap_lon,
            "mp_number": e.get("mp_number"),
        }

        # ----------------------------------------------------
        # CASE 1:
        # No extracted coordinates.
        # ----------------------------------------------------

        if cap_lat in (None, "") or cap_lon in (None, ""):

            results.append({
                **base,
                "master_latitude": None,
                "master_longitude": None,
                "distance_m": None,
                "status": "Missing Coordinates",
                "duplicate_id_in_master": False,
            })

            continue

        # ----------------------------------------------------
        # Convert captured coordinates to float.
        # ----------------------------------------------------

        try:

            cap_lat_f = float(cap_lat)
            cap_lon_f = float(cap_lon)

        except (ValueError, TypeError):

            results.append({
                **base,
                "master_latitude": None,
                "master_longitude": None,
                "distance_m": None,
                "status": "Invalid Coordinate Format",
                "duplicate_id_in_master": False,
            })

            continue

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # EXACT MATCH ONLY against master_site_id.
        #
        # This means:
        #
        #     1683
        #
        # will match:
        #
        #     1683
        #
        # but NOT:
        #
        #     CI-1683
        #     28.1683
        #     71.91683
        #     34.01683
        #
        # because those are not in master_site_id as an
        # exact value.
        # ----------------------------------------------------

        match = master_df[
            master_df["master_site_id"] == site_id
        ]

        # ----------------------------------------------------
        # CASE 2:
        # Site ID does not exist in master.
        # ----------------------------------------------------

        if match.empty:

            results.append({
                **base,
                "master_latitude": None,
                "master_longitude": None,
                "distance_m": None,
                "status": "Unknown Site",
                "duplicate_id_in_master": False,
            })

            continue

        # ----------------------------------------------------
        # Was THIS Site ID duplicated inside the master?
        # ----------------------------------------------------

        is_duplicate_in_master = (
            site_id in master_duplicate_ids
        )

        # ----------------------------------------------------
        # Find the closest matching master row.
        #
        # This works for BOTH:
        #
        #   one master row
        #
        # and:
        #
        #   multiple master rows with the same Site ID.
        # ----------------------------------------------------

        best_dist = None
        best_lat = None
        best_lon = None

        for _, candidate in match.iterrows():

            try:

                master_lat = float(
                    candidate["master_latitude"]
                )

                master_lon = float(
                    candidate["master_longitude"]
                )

            except (ValueError, TypeError):

                continue

            # ------------------------------------------------
            # Calculate distance using the actual coordinates.
            #
            # No duplicate rejection occurs here.
            # ------------------------------------------------

            distance = haversine_m(
                cap_lat_f,
                cap_lon_f,
                master_lat,
                master_lon,
            )

            if best_dist is None or distance < best_dist:

                best_dist = distance
                best_lat = master_lat
                best_lon = master_lon

        # ----------------------------------------------------
        # No usable master coordinates.
        # ----------------------------------------------------

        if best_dist is None:

            results.append({
                **base,
                "master_latitude": None,
                "master_longitude": None,
                "distance_m": None,
                "status": "Invalid Coordinate Format",
                "duplicate_id_in_master": is_duplicate_in_master,
            })

            continue

        # ----------------------------------------------------
        # Distance status.
        #
        # IMPORTANT:
        # Duplicate master IDs do NOT alter the distance status.
        #
        # Example:
        #
        #     142.9m
        #
        # remains:
        #
        #     Within 300m
        #
        # even if the master has multiple rows for that ID.
        # ----------------------------------------------------

        status = (
            "Within 300m"
            if best_dist <= MAX_M
            else "Outside 300m"
        )

        results.append({
            **base,
            "master_latitude": best_lat,
            "master_longitude": best_lon,
            "distance_m": round(best_dist, 1),
            "status": status,
            "duplicate_id_in_master": is_duplicate_in_master,
        })

    return results