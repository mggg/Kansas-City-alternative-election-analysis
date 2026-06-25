"""
Kansas City VTDs Data Generator
Generates geopackage of VTDs for Kansas City using spatial join and clip
"""

import geopandas as gpd
import pandas as pd
import os
from pathlib import Path


def load_data(gpkg_path, kc_boundary_geoid="2938000"):
    """
    Load Missouri geopackage and KC boundary
    
    Args:
        gpkg_path: Path to Peter's geopackage
        kc_boundary_geoid: GEOID of Kansas City (default: 2938000)
    
    Returns:
        mo_vtds, kc_boundary: GeoDataFrames
    """
    
    # Read Peter's geopackage
    print(f"  Reading {gpkg_path}...")
    mo_vtds = gpd.read_file(gpkg_path, layer="mo_districtr_vtd_view_v1").to_crs(epsg=4326)
    print(f"{len(mo_vtds)} VTDs in Missouri")
    
    # Download census places shapefile
    url_places = "https://www2.census.gov/geo/tiger/TIGER2020/PLACE/tl_2020_29_place.zip"
    places = gpd.read_file(url_places)
    
    # Filter Kansas City
    kc_boundary = places[places["GEOID"] == kc_boundary_geoid].copy()
    if len(kc_boundary) == 0:
        raise ValueError(f"Kansas City with GEOID {kc_boundary_geoid} not found")
    
    kc_name = kc_boundary["NAME"].values[0]
    print(f"Boundary of {kc_name}")
    
    return mo_vtds, kc_boundary

def clip_and_filter(mo_vtds, kc_boundary, min_area_pct=0.01):
    """
    Clip VTDs to KC boundary and filter by minimum area
    
    Args:
        mo_vtds: GeoDataFrame of Missouri VTDs
        kc_boundary: GeoDataFrame of KC boundary
        min_area_pct: Minimum area as percentage of median original (default: 1%)
    
    Returns:
        kc_vtds_filtered: Filtered GeoDataFrame of VTDs
    """
    
    # Project to UTM for precision
    proj_crs = mo_vtds.estimate_utm_crs()
    mo_vtds_utm = mo_vtds.to_crs(proj_crs)
    kc_boundary_utm = kc_boundary.to_crs(proj_crs)
    
    # Clip
    kc_vtds_clip = gpd.clip(mo_vtds_utm, kc_boundary_utm).to_crs(epsg=4326)
    print(f"{len(kc_vtds_clip)} VTDs after clip")
    
    # Filter by minimum area
    original_areas = mo_vtds_utm.geometry.area
    min_area = original_areas.median() * min_area_pct
    
    kc_vtds_filtered = kc_vtds_clip[kc_vtds_clip.geometry.area > min_area].copy()
    removed = len(kc_vtds_clip) - len(kc_vtds_filtered)
    print(f"{len(kc_vtds_filtered)} VTDs after filtering")
    print(f"{removed} VTDs removed (area < {min_area_pct*100:.0f}%)")
    
    return kc_vtds_filtered


def print_statistics(kc_vtds):
    """
    Print dataset statistics
    """
    print("\n" + "=" * 60)
    print("KANSAS CITY STATISTICS")
    print("=" * 60)
    
    # Project to UTM for correct area calculation
    kc_vtds_utm = kc_vtds.to_crs("EPSG:26915")
    
    print(f"\n Geography:")
    print(f"  Total VTDs: {len(kc_vtds)}")
    print(f"  Total area: {kc_vtds_utm.geometry.area.sum() / 1e6:.2f} km²")
    print(f"  Average area per VTD: {kc_vtds_utm.geometry.area.mean() / 1e6:.4f} km²")
    
    print(f"\n Total population (total_pop_20):")
    print(f"  Total: {kc_vtds['total_pop_20'].sum():,.0f}")
    print(f"  Average per VTD: {kc_vtds['total_pop_20'].mean():,.0f}")
    
    print(f"\n VAP (Voting Age Population):")
    print(f"  Total VAP: {kc_vtds['total_vap_20'].sum():,.0f}")
    print(f"  % of population: {(kc_vtds['total_vap_20'].sum() / kc_vtds['total_pop_20'].sum() * 100):.1f}%")
    
    print(f"\n Demographics (VAP by race/ethnicity):")
    demo_cols = ['bvap_20', 'hvap_20', 'white_vap_20', 'asian_nhpi_vap_20', 'amin_vap_20', 'other_vap_20']
    for col in demo_cols:
        if col in kc_vtds.columns:
            total = kc_vtds[col].sum()
            pct = (total / kc_vtds['total_vap_20'].sum() * 100)
            print(f"  {col:20s}: {total:10,.0f} ({pct:5.1f}%)")
    
    if 'pres_20_dem' in kc_vtds.columns and 'pres_20_rep' in kc_vtds.columns:
        print(f"\n Electoral results (2020 Presidential):")
        dem = kc_vtds['pres_20_dem'].sum()
        rep = kc_vtds['pres_20_rep'].sum()
        total = dem + rep
        print(f"  Democrats: {dem:,.0f} ({dem/total*100:.1f}%)")
        print(f"  Republicans: {rep:,.0f} ({rep/total*100:.1f}%)")
        print(f"  Total votes: {total:,.0f}")
    
    print()

def validate_geometries(gdf):
    """
    Validate and fix geometries for GerryChain
    
    Args:
        gdf: GeoDataFrame of VTDs
    
    Returns:
        gdf: Validated GeoDataFrame
    """
    print("\nValidating geometries...")
    
    # Fix invalid geometries
    invalids = (~gdf.geometry.is_valid).sum()
    if invalids > 0:
        print(f"  Fixing {invalids} invalid geometries...")
        gdf["geometry"] = gdf.geometry.make_valid()
    
    # Remove empty geometries
    empties = gdf.geometry.is_empty.sum()
    if empties > 0:
        print(f"  Removing {empties} empty geometries...")
        gdf = gdf[~gdf.geometry.is_empty].copy()
    
    print(f"  ✓ {len(gdf)} valid geometries")
    return gdf

def export_geojson(kc_vtds, output_path):
    """
    Export to Geopackage (.gpkg)
    
    Args:
        kc_vtds: GeoDataFrame of KC VTDs
        output_path: Output file path
    """
    # Create directory if not exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to GPKG
    kc_vtds.to_file(output_path, driver="GPKG")

def main():
    """Main workflow"""
    
    print("=" * 60)
    print("Kansas City VTDs Data Generator")
    print("=" * 60)
    
    # Paths
    gpkg_path = "data/BETA_release_packages/mo_districtr_vtd_view_v1.gpkg"
    output_path = "data/kcmo_districts_vtd.gpkg"
    
    # Load data
    mo_vtds, kc_boundary = load_data(gpkg_path)
    
    # Clip and filter
    kc_vtds_filtered = clip_and_filter(mo_vtds, kc_boundary)
    
    # Statistics
    print_statistics(kc_vtds_filtered)
    
    # Validate geometries
    kc_vtds_filtered = validate_geometries(kc_vtds_filtered)
    
    # Export
    export_geojson(kc_vtds_filtered, output_path)

if __name__ == "__main__":
    main()
