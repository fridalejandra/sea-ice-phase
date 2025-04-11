import xarray as xr

# ---- INPUT FILES ---- #
base_file = "/Users/fridaperez/Developer/repos/phase_project/Stammerjohn_2008/merged_bootsstrap.nc"
new_file = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_extended_SH.nc"

# ---- OUTPUT FILE ---- #
final_merged_file = "/data/bootstrap_smmr/merged_bootstrap_SH_1979_06302024.nc"

# ---- LOAD ---- #
ds_base = xr.open_dataset(base_file)
ds_new = xr.open_dataset(new_file)

# ---- Match variable name ---- #
var_base = [v for v in ds_base.data_vars if v.endswith("_ICECON")][0]
var_new = [v for v in ds_new.data_vars if v.endswith("_ICECON")][0]

if var_base != var_new:
    print(f"⚠️ Variable names differ: {var_base} vs {var_new}. Renaming...")
    ds_new = ds_new.rename({var_new: var_base})

# ---- MERGE & SORT ---- #
merged = xr.concat([ds_base[var_base], ds_new[var_base]], dim="time")
merged = merged.sortby("time")

# ---- SAVE ---- #
merged.to_dataset(name=var_base).to_netcdf(final_merged_file)
print(f"✅ Final merged file written to: {final_merged_file}")
