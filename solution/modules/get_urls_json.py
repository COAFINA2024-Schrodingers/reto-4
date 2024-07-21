# Importar librerias necesarias
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import nest_asyncio
import requests

nest_asyncio.apply()

import json
# Guardar como archivo json cada carpeta de año
def save_as_json(file, dictionary):
    with open("./years_512/" + file + ".json", 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
        
        
# URL del servidor SOHO donde están las imágenes
BASE_URL = "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"

# Años, filtros
years = [ str(year) for year in range(1996,2025)]
# filters = ['c2', 'c3', 'eit171', 'eit195', 'eit284', 'eit304', 'mdiigr', 'mdimag']
filters_1996 = ['mdiigr', 'mdimag']
filters_2011 = ['hmiigr', 'hmimag']


# Obtener URLs sin paralelizar, sin forma asíncrona
def get_urls(page_url, look=''):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    urls = []
    hrefs = []

    for link in soup.find_all('a', href=True):
        href = link['href']
        if look in href:
            urls.append(page_url + href)
            hrefs.append(href)
    return urls, hrefs


# Obtiene las horas en las que las imágenes fueron tomadas a partir de los nombres de las imágenes
def get_hours(file_name):
    hours = file_name.split("_")
    return hours[1]

# Obtiene la resolución de las imágenes a partir del nombre de la imagen
def get_res(file_name):
    resolution = file_name.split("_")
    return resolution[3]


# Obtiene la URL de las imágenes filtrado por resolución
async def get_image_urls(url, session, resolution):
    retries = 10
    attempt = 0
    while attempt < retries:
        try:
            async with session.get(url=url) as response:
                html = await response.read()

                # Analizar el html con BS
                urls = [] # Lista de urls de las imágenes
                hrefs = [] # Lista de nombres de las imágenes
                soup = BeautifulSoup(html, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    #Filtra aquellos que terminen en .jpg y tengan cierta resolución
                    if href.endswith('.jpg') and get_res(href) == resolution + ".jpg":
                        hrefs.append(href)

                # Selecciona aquellas imágenes que hayan sido tomadas cerca de las 12 pm y añade las urls de ellas a la lista
                hours = [ int(get_hours(href)) for href in  hrefs]    
                close_to_12 = [ abs(1200 - hour)  for hour in hours ]
        
                for href, close in zip(hrefs, close_to_12):
                    if min(close_to_12) == close:
                        urls.append( url + href)
                
                return urls

        except Exception as e:
            attempt += 1
            # print(f"Unable to get url {url} due to {e.__class__}.")
            pass
    print(f"Failed to get url {url} after {retries} attempts.")
    return None


# Accede a las URLs de manera asincrona y devuelve para cada url las urls de las imágenes de adentro
async def main(urls, resolution='512'):
    async with aiohttp.ClientSession() as session:
        result = await asyncio.gather(*(get_image_urls(url, session, resolution) for url in urls))
    # print(f"Finally all. Return is a list of len {len(result)} outputs.")
    print("Done.")
    return result


def get_dict_filter(filter_dict_):
    new_filter_dict = filter_dict_.copy()

    for filter_ in list(filter_dict_.keys()):
        all_images = []

        for values_date in list(filter_dict_[filter_].values()):
            all_images.extend(values_date)
        new_filter_dict[filter_] = all_images

    return new_filter_dict



year_dict = dict()

# Itera sobre los años
def get_json_years():
    for year in years[16:17]:
        filter_dict = dict() # Se guardarán los filtros como diccionario
        # Itera sobre los filtros
        for filter in filters_2011: 
            filter_url = BASE_URL + year + "/" + filter + "/" # Url de la carpeta de un filtro

            date_urls = get_urls(filter_url)[0][5::] # Obtiene las urls de las fechas 
            dates = [ date[-9:].replace("/","") for date in date_urls] # Filra las ulrs para obtener solo fechas
            a = asyncio.run(main(date_urls)) # Obtiene las urls de las imágenes en listas
            date_dict = dict(zip(dates, a)) # Genera un diccionario de fechas con fecha:lista_de_imágenes

            filter_dict[filter] = date_dict # Añade el diccionario de fechas al diccionario de filtros

        year_dict[year] = filter_dict   # Añade el diccionario de filtros al diccionario de años


        # save_as_json(year, year_dict)
        save_as_json(year, get_dict_filter(filter_dict))
