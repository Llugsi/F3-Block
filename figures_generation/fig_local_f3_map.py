# [Script 18] Single-column (3.5") localized acquisition orientation layout
import matplotlib.pyplot as plt
import numpy as np

# 1. Configurar dimensiones cuadradas perfectas para 1 columna IEEE (3.5 x 3.5 pulgadas)
fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

# Ventana de coordenadas de la Malla Base Regional
lon_min, lon_max = 2.0, 8.0
lat_min, lat_max = 52.0, 57.0

# Ampliamos los márgenes para garantizar espacio libre abajo para las leyendas
ax.set_xlim(lon_min - 1.5, lon_max + 1.5)
ax.set_ylim(lat_min - 1.6, lat_max + 0.6)

# 2. VÉRTICES DE LA COSTA: Perfil real de Europa Continental
europe_coast_lon = [
    2.5, 3.2, 3.6, 4.0, 4.4, 4.7, 4.9, 5.3, 5.8, 6.4, 7.0, 7.8, 8.6, 
    8.9, 8.6, 8.1, 8.4, 8.6, 8.4, 8.1, 8.1, 8.3, 8.5, 8.2, 8.1, 8.2, 8.4, 8.6
]
europe_coast_lat = [
    51.1, 51.4, 51.7, 52.1, 52.5, 52.9, 53.1, 53.3, 53.4, 53.5, 53.6, 53.7, 53.9, 
    54.1, 54.4, 54.6, 54.8, 55.1, 55.4, 55.7, 56.1, 56.4, 56.7, 57.0, 57.2, 57.4, 57.5, 57.7
]

# Dibujar la costa continental (Hilo gris técnico delgado)
ax.plot(europe_coast_lon, europe_coast_lat, color='#7f8c8d', linestyle='-', linewidth=0.8, zorder=1)

# Rotulaciones geográficas institucionales limpias
ax.text(6.8, 52.4, 'Netherlands\n& Germany', color='#95a5a6', fontsize=7, weight='bold', ha='center', va='center')
ax.text(3.2, 52.7, 'F3 Regional\nAsset Grid', color='#2980b9', fontsize=8, style='italic', ha='center', va='center')

# 3. MÁSCARA DE LA GRILLA REGIONAL (Línea discontinua técnica)
bbox = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, 
                     edgecolor='#7f8c8d', facecolor='none', linewidth=1.2, linestyle='--', zorder=5)
ax.add_patch(bbox)

# Generar mallado fino interno
x_grid = np.linspace(lon_min - 4, lon_max + 4, 25)
for x in x_grid:
    t = np.linspace(0, 1, 100)
    
    # Perfiles Inline (Pendiente positiva)
    x_in = x + t * 6
    y_in = lat_min - 1 + t * 7
    mask_in = (x_in >= lon_min) & (x_in <= lon_max) & (y_in >= lat_min) & (y_in <= lat_max)
    if np.any(mask_in):
        ax.plot(x_in[mask_in], y_in[mask_in], color='#3b5998', linewidth=0.3, alpha=0.18, zorder=2)
        
    # Perfiles Crossline (Pendiente negativa)
    x_cr = x - t * 6
    y_cr = lat_min - 1 + t * 7
    mask_cr = (x_cr >= lon_min) & (x_cr <= lon_max) & (y_cr >= lat_min) & (y_cr <= lat_max)
    if np.any(mask_cr):
        ax.plot(x_cr[mask_cr], y_cr[mask_cr], color='#3b5998', linewidth=0.3, alpha=0.18, zorder=2)

# ==========================================
# 3.5. UBICACIÓN DEL BLOQUE SÍSMICO F3 REAL
# ==========================================
lon_f3_min, lat_f3_min = 4.64, 54.81  
box_w, box_h = 0.25, 0.15  

f3_box = plt.Rectangle((lon_f3_min, lat_f3_min), box_w, box_h, 
                       edgecolor='#c0392b', facecolor='none', linewidth=2.0, zorder=12)
ax.add_patch(f3_box)

# Texto descriptivo arriba a la izquierda del cruce de vectores
ax.text(3.6, 55.6, 'F3 Block\nTarget Area', 
        color='#c0392b', fontsize=8, weight='bold', va='center', ha='center')

# ==========================================
# 4. TRAZADO DE VECTORES SÍSMICOS
# ==========================================
center_lon, center_lat = lon_f3_min + box_w/2, lat_f3_min + box_h/2

# Perfil Inline (Flecha Roja)
ax.annotate('', xy=(center_lon + 1.2, center_lat + 0.8), xytext=(center_lon - 1.2, center_lat - 0.8),
            arrowprops=dict(arrowstyle="->", color="#cc1111", lw=1.2, mutation_scale=8), zorder=10)

# Perfil Crossline (Flecha Verde)
ax.annotate('', xy=(center_lon + 1.2, center_lat - 0.8), xytext=(center_lon - 1.2, center_lat + 0.8),
            arrowprops=dict(arrowstyle="->", color="#228b22", lw=1.2, mutation_scale=8), zorder=10)

# 5. ROTULACIÓN DE COORDENADAS EN LAS ESQUINAS (8pt IEEE)
ax.text(lon_min - 0.1, lat_min - 0.20, r'$2^\circ\mathrm{E}$', fontsize=8, color='black', weight='bold', ha='right')
ax.text(lon_min - 0.1, lat_min + 0.05, r'$52^\circ\mathrm{N}$', fontsize=8, color='black', weight='bold', ha='right')

ax.text(lon_max + 0.1, lat_max + 0.10, r'$8^\circ\mathrm{E}$', fontsize=8, color='black', weight='bold', ha='left')
ax.text(lon_max + 0.1, lat_max - 0.15, r'$57^\circ\mathrm{N}$', fontsize=8, color='black', weight='bold', ha='left')

# ==========================================
# 6. LEYENDAS FORMALES CON SEPARACIÓN ULTRA-ANCHA COREGIDA
# ==========================================
# Nivel vertical de la leyenda (bajado a lat_min - 1.2 para evitar choques)
leyenda_y = lat_min - 1.2

# Componente Inline (Lado Izquierdo: inicia en x=1.0)
ax.plot([1.0, 1.5], [leyenda_y, leyenda_y], color='#cc1111', lw=1.5)
ax.text(1.6, leyenda_y, 'Inline Direction', color='#cc1111', fontsize=8, weight='bold', va='center', ha='left')

# Componente Crossline (Lado Derecho: inicia en x=5.2)
ax.plot([5.2, 5.7], [leyenda_y, leyenda_y], color='#228b22', lw=1.5)
ax.text(5.8, leyenda_y, 'Crossline Direction', color='#228b22', fontsize=8, weight='bold', va='center', ha='left')

# Ocultar los ejes convencionales
ax.axis('off')

# Guardar la imagen optimizada
plt.savefig('ieee_col_north_sea_map.png', format='png', bbox_inches='tight', transparent=True)
plt.show()
