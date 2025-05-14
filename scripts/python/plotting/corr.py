# === LOAD OUTPUTS === #
r_map = np.load(os.path.join(SAVE_DIR, "r_map.npy"))
r2_map = np.load(os.path.join(SAVE_DIR, "r2_map.npy"))
resid_map = np.load(os.path.join(SAVE_DIR, "resid_map.npy"))


# === PLOT FUNCTION === #
def plot_map(data, title, cmap, vmin, vmax, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(
        advance.x, advance.y, data,
        transform=ccrs.SouthPolarStereo(),
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label(title)

    plt.title(title, fontsize=14)
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


# === GENERATE PLOTS === #
plot_map(r_map, "Correlation (r) between Retreat and Advance", cmap="coolwarm", vmin=-1, vmax=1,
         filename="correlation_r.png")
plot_map(r2_map, "Predictive Skill (R²) Retreat → Advance", cmap="viridis", vmin=0, vmax=1,
         filename="predictive_skill_r2.png")
plot_map(resid_map, "Residual Standard Deviation", cmap="magma", vmin=0, vmax=np.nanmax(resid_map),
         filename="residual_std.png")

print("✅ Plotting complete.")
