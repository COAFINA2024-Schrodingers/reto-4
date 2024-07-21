import aiohttp
import asyncio
import nest_asyncio
import numpy as np
import json
import cv2

nest_asyncio.apply()


def get_data_from_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def join_years_data(years_data, filter_dict):
    for year in years_data:
        for filter_ in list(filter_dict.keys()):
            filter_dict[filter_].extend(year[filter_])
    # return filter_dict
    
def get_date(url):
    list_url = url.split("/")
    return list_url[8]
    
    
years_file = ["./years_512/" + str(year) + ".json" for year in range(1996,2025)]
years_data_1996 = [ get_data_from_json(year) for year in years_file[0:15] ]
years_data_2011 = [ get_data_from_json(year) for year in years_file[15:24] ]
filter_dict_1996 = {'mdiigr':[], 'mdimag':[]}
filter_dict_2011 = {'hmiigr':[], 'hmimag':[]}

join_years_data(years_data_1996, filter_dict_1996)
join_years_data(years_data_2011, filter_dict_2011)


def get_per_year(data):
    data_to_use = data.copy()
    # Obtiene la URL
    async def get_array_image(url, session):
        try:
            async with session.get(url=url) as response:
                response = await response.read()
                
                image_array = np.frombuffer(response, np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return (get_date(url), img)

        except Exception as e:
            # print(f"Unable to get url {url} due to {e.__class__}.")
            pass

    # Accede a las URLs de manera asincrona y devuelve para cada url las urls de las imágenes de adentro
    async def main(urls):
        async with aiohttp.ClientSession() as session:
            ret = await asyncio.gather(*(get_array_image(url, session) for url in urls))
        # print(f"Finally all. Return is a list of len {len(ret)} outputs.")
        return ret

    for filter in list(data.keys()):
        a = asyncio.run(main(data[filter]))
        data_to_use[filter] = a
    return data_to_use