# [Script 19] Macro geographic context layout with supersampled poly coastlines
import matplotlib.pyplot as plt
import numpy as np

# 1. Configure high-resolution plot for IEEE single-column layouts (Width: 3.5 inches)
fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

# Precise North Sea geographic window with optimized margins
lon_min, lon_max = -5.0, 11.0
lat_min, lat_max = 49.0, 62.0
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# 2. GEOGRAPHIC BASELINE REPOSITORIES (Offline boundary coordinates)
raw_continental = [
    [-5.0, 48.0], [-4.8, 48.3], [-4.4, 48.5], [-3.8, 48.2], [-3.0, 48.6], [-2.5, 48.6], [-1.8, 48.8],
    [-1.6, 49.3], [-1.0, 49.4], [-0.5, 49.3], [0.1, 49.5], [0.7, 49.4], [1.3, 50.1], [1.6, 50.3],
    [1.9, 50.7], [2.2, 51.0], [2.6, 51.1], [3.2, 51.3], [3.6, 51.5], [3.9, 51.8], [4.1, 52.2], 
    [4.4, 52.3], [4.5, 52.5], [4.7, 52.9], [5.0, 53.1], [5.3, 53.2], [5.7, 53.3], [6.1, 53.4], 
    [6.5, 53.5], [6.9, 53.4], [7.4, 53.6], [7.9, 53.6], [8.3, 53.8], [8.6, 53.9], [8.9, 54.1], 
    [8.8, 54.3], [8.6, 54.4], [8.3, 54.5], [8.1, 54.6], [8.2, 54.7], [8.4, 54.8], [8.6, 55.0], 
    [8.6, 55.1], [8.5, 55.3], [8.4, 55.4], [8.2, 55.6], [8.1, 55.7], [8.1, 55.9], [8.1, 56.1], 
    [8.2, 56.3], [8.3, 56.4], [8.4, 56.6], [8.5, 56.7], [8.4, 56.9], [8.2, 57.1], [8.3, 57.2], 
    [8.4, 57.3], [8.5, 57.5], [8.6, 57.6], [9.0, 57.7], [9.5, 57.8], [10.0, 57.8], [10.5, 57.7], 
    [10.6, 57.3], [10.6, 56.8], [10.4, 56.5], [10.1, 56.1], [10.1, 55.9], [10.3, 55.7], [10.6, 55.6], 
    [10.9, 55.5], [11.2, 55.6], [11.5, 55.7], [11.4, 55.2], [11.1, 54.6], [11.0, 54.4], [10.8, 54.3], 
    [10.4, 54.3], [9.9, 54.3], [9.4, 54.5], [8.9, 54.8], [8.5, 54.5], [8.0, 54.0], [7.5, 53.8], 
    [7.0, 53.5], [6.3, 52.5], [5.5, 51.5], [4.5, 51.2], [3.5, 51.0], [2.5, 50.2], [1.5, 49.5], 
    [0.5, 49.2], [-1.0, 49.1], [-2.0, 49.0], [-3.5, 48.5], [-5.0, 48.0]
]

raw_uk = [
    [-5.0, 50.1], [-4.5, 50.2], [-4.0, 50.4], [-3.5, 50.4], [-3.0, 50.5], [-2.5, 50.6], [-2.0, 50.6], 
    [-1.5, 50.7], [-1.0, 50.7], [-0.5, 50.8], [0.0, 50.9], [0.5, 51.1], [1.0, 51.2], [1.3, 51.3], 
    [1.4, 51.6], [1.4, 51.9], [1.2, 52.0], [1.0, 52.0], [1.1, 52.2], [1.3, 52.3], [1.6, 52.5], 
    [1.7, 52.7], [1.6, 52.9], [1.3, 53.0], [1.0, 53.1], [0.6, 53.3], [0.2, 53.6], [-0.2, 54.0], 
    [-0.5, 54.3], [-0.9, 54.5], [-1.2, 54.7], [-1.4, 55.0], [-1.5, 55.2], [-2.0, 55.5], [-2.6, 55.7], 
    [-2.3, 55.9], [-2.0, 56.0], [-2.3, 56.2], [-2.6, 56.4], [-2.4, 56.7], [-2.0, 57.0], [-1.9, 57.3], 
    [-1.8, 57.6], [-2.3, 57.7], [-3.0, 57.7], [-3.2, 58.0], [-3.4, 58.2], [-3.9, 58.4], [-4.3, 58.6], 
    [-4.8, 58.6], [-5.0, 58.6], [-4.8, 58.3], [-4.5, 58.0], [-4.9, 57.7], [-5.2, 57.5], [-5.5, 57.3], 
    [-5.7, 57.0], [-5.2, 56.7], [-4.8, 56.5], [-5.2, 56.3], [-5.6, 56.0], [-5.2, 55.7], [-4.8, 55.4], 
    [-5.1, 55.4], [-5.5, 55.4], [-5.2, 55.1], [-4.8, 54.8], [-4.2, 54.8], [-3.5, 54.8], [-3.2, 54.2], 
    [-3.0, 53.5], [-3.6, 53.4], [-4.1, 53.3], [-4.5, 53.4], [-4.9, 53.2], [-5.2, 52.9], [-4.6, 52.6], 
    [-4.0, 52.3], [-4.15, 52.15], [-4.3, 52.0], [-4.4, 51.8], [-4.5, 51.6], [-4.7, 51.35], [-4.9, 51.1], 
    [-5.05, 50.85], [-5.2, 50.6], [-5.15, 50.45], [-5.1, 50.3], [-5.05, 50.2], [-5.0, 50.1]
]

raw_norway = [
    [5.0, 62.0], [5.3, 61.8], [5.5, 61.5], [5.2, 61.2], [4.8, 61.0], [5.0, 60.7], [5.2, 60.5], 
    [5.1, 60.2], [5.0, 60.0], [5.2, 59.7], [5.4, 59.4], [5.6, 59.2], [5.8, 59.0], [6.0, 58.7], 
    [6.3, 58.5], [6.6, 58.2], [7.0, 58.0], [7.5, 58.0], [8.0, 58.1], [8.3, 58.2], [8.6, 58.3], 
    [9.0, 58.6], [9.5, 58.9], [10.0, 59.1], [10.5, 59.3], [10.8, 59.2], [11.0, 59.0], [11.0, 58.6], 
    [10.9, 58.2], [10.9, 57.8], [11.0, 57.5], [11.2, 57.0], [11.5, 56.5], [12.0, 56.2], [12.5, 56.0], 
    [12.8, 55.7], [13.0, 55.4], [13.5, 55.3], [14.0, 55.2], [14.0, 62.0], [5.0, 62.0]
]

# 3. POLYGON SUPERSAMPLING AND LINEAR INTERPOLATION SMOOTHING ALGORITHM
def supersample_polygon(poly, factor=6):
    poly = np.array(poly)
    x, y = poly[:, 0], poly[:, 1]
    new_x, new_y = [], []
    for i in range(len(poly)-1):
        xs = np.linspace(x[i], x[i+1], factor, endpoint=False)
        ys = np.linspace(y[i], y[i+1], factor, endpoint=False)
        new_x.extend(xs)
        new_y.extend(ys)
    new_x.append(x[-1])
    new_y.append(y[-1])
    return np.column_stack((new_x, new_y))

# Automatic generation of high-density, smooth cartographic vertices
continental_ultra = supersample_polygon(raw_continental, factor=6)
uk_ultra = supersample_polygon(raw_uk, factor=6)
norway_ultra = supersample_polygon(raw_norway, factor=6)

# Render solid filled polygons eliminating rigid edge boundaries
ax.fill(continental_ultra[:, 0], continental_ultra[:, 1], facecolor='#f5f6fa', edgecolor='#bdc3c7', linewidth=0.4, zorder=1)
ax.fill(uk_ultra[:, 0], uk_ultra[:, 1], facecolor='#f5f6fa', edgecolor='#bdc3c7', linewidth=0.4, zorder=1)
ax.fill(norway_ultra[:, 0], norway_ultra[:, 1], facecolor='#f5f6fa', edgecolor='#bdc3c7', linewidth=0.4, zorder=1)

# Cartographic nomenclature labeling
ax.text(-2.5, 53.5, 'United\nKingdom', color='#7f8c8d', fontsize=8, ha='center', va='center', style='italic')
ax.text(7.5, 51.5, 'Continental\nEurope', color='#7f8c8d', fontsize=8, ha='center', va='center', style='italic')
ax.text(8.5, 60.5, 'Norway', color='#7f8c8d', fontsize=8, ha='center', va='center', style='italic')
ax.text(0.0, 57.5, 'North Sea', color='#2980b9', fontsize=10, weight='bold', ha='center', va='center')

# 4. HIGH-DENSITY MESH OVERLAY (Decoupled tracking stripes)
grid_space = np.linspace(lon_min - 10, lon_max + 10, 42)
for g in grid_space:
    t = np.linspace(0, 1, 100)
    x_in = g + t * 15
    y_in = lat_min - 2 + t * 16
    mask_in = (x_in >= lon_min) & (x_in <= lon_max) & (y_in >= lat_min) & (y_in <= lat_max)
    ax.plot(x_in[mask_in], y_in[mask_in], color='#3b5998', linewidth=0.3, alpha=0.28, zorder=2)
    
    x_cr = g - t * 15
    mask_cr = (x_cr >= lon_min) & (x_cr <= lon_max) & (y_in >= lat_min) & (y_in <= lat_max)
    ax.plot(x_cr[mask_cr], y_in[mask_cr], color='#3b5998', linewidth=0.3, alpha=0.28, zorder=2)

# ==========================================
# 4.5. SURVEY GEOMETRY: REGIONAL ASSET GRID MASK (2°-8°E, 52°-57°N)
# ==========================================
grid_box = plt.Rectangle((2.0, 52.0), 6.0, 5.0, 
                         edgecolor='#7f8c8d', facecolor='none', linewidth=1.0, linestyle='--', zorder=5)
ax.add_patch(grid_box)
ax.text(2.2, 52.3, 'F3 Asset Grid', color='#535c68', fontsize=7, weight='bold', ha='left', va='bottom')

# ==========================================
# 5. FOCUS AREA BBOX: Localized F3 Block Target (Geodetic Coordinates)
# ==========================================
lon_f3_min, lat_f3_min = 4.64, 54.81  
box_w, box_h = 0.25, 0.15  

study_box = plt.Rectangle((lon_f3_min, lat_f3_min), box_w, box_h, 
                          edgecolor='#c0392b', facecolor='none', linewidth=1.5, zorder=10)
ax.add_patch(study_box)

# Bounding descriptor positioned to the right for maximum tracking scannability
ax.text(lon_f3_min + 0.5, lat_f3_min, 'F3 Block\nTarget', 
        color='#c0392b', fontsize=8, weight='bold', va='center', ha='left')

# 6. GEODETIC AXES FORMATTING 
ax.set_xticks([-4, 0, 4, 8, 12])
ax.set_yticks(range(50, 63, 2))
ax.set_xticklabels([r'$4^\circ\mathrm{W}$', r'$0^\circ$', r'$4^\circ\mathrm{E}$', r'$8^\circ\mathrm{E}$', r'$12^\circ\mathrm{E}$'], fontsize=8)
ax.set_yticklabels([r'$50^\circ\mathrm{N}$', r'$52^\circ\mathrm{N}$', r'$54^\circ\mathrm{N}$', r'$56^\circ\mathrm{N}$', r'$58^\circ\mathrm{N}$', r'$60^\circ\mathrm{N}$', r'$62^\circ\mathrm{N}$'], fontsize=8)

# Subtle background grid configuration
ax.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
ax.tick_params(axis='both', which='major', labelsize=8)

# Save publication-ready layout optimized for LaTeX injection
plt.savefig('ieee_regional_f3_perfect.png', format='png', bbox_inches='tight', transparent=True)
plt.show()
