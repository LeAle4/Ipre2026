import rasterio
from pyproj import Geod

def obtener_resolucion_cm(ruta_tif):
    with rasterio.open(ruta_tif) as src:
        res_x, res_y = src.res
        crs = src.crs

        # 1. Si el archivo ya está en un sistema proyectado (Metros, Pies, etc.)
        if crs.is_projected:
            # Obtenemos el factor de conversión a metros (p.ej. pies -> metros)
            nombre_unidad, factor_a_metros = crs.linear_units_factor
            res_x_cm = res_x * factor_a_metros * 100
            res_y_cm = res_y * factor_a_metros * 100
            metodo = f"Proyectado ({nombre_unidad})"

        # 2. Si el archivo está en grados (Geográfico: WGS84, etc.)
        elif crs.is_geographic:
            # Calculamos la distancia real en el centro de la imagen
            # ya que 1 grado no mide lo mismo en el ecuador que en los polos.
            lon_centro = (src.bounds.left + src.bounds.right) / 2
            lat_centro = (src.bounds.top + src.bounds.bottom) / 2
            
            geod = Geod(ellps='WGS84')
            
            # Calculamos distancia horizontal (X)
            _, _, dist_x_m = geod.inv(lon_centro, lat_centro, lon_centro + res_x, lat_centro)
            # Calculamos distancia vertical (Y)
            _, _, dist_y_m = geod.inv(lon_centro, lat_centro, lon_centro, lat_centro + res_y)
            
            res_x_cm = dist_x_m * 100
            res_y_cm = dist_y_m * 100
            metodo = "Geográfico (Grados convertidos a CM)"
        
        else:
            # Caso genérico / Desconocido (asumimos metros por defecto)
            res_x_cm, res_y_cm = res_x * 100, res_y * 100
            metodo = "Desconocido (Asumiendo metros)"

        print(f"--- Análisis de: {ruta_tif} ---")
        print(f"Método: {metodo}")
        print(f"Resolución X: {res_x_cm:.2f} cm/px")
        print(f"Resolución Y: {res_y_cm:.2f} cm/px")
        print("-" * 30)

obtener_resolucion_cm('tu_archivo.tif')