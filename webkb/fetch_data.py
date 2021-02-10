from webkb import get_dataset


for data in ["Cornell", 'Texas', "Wisconsin", "Squirrel", "Chameleon", "Film"]:
    print('fetching '+ data)
    get_dataset(data)

