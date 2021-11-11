from .datasets import get_dataset


for data in ["Cornell", 'Texas', "Wisconsin", "Squirrel", "Chameleon", "Film"]:
    print('fetching '+ data)
    get_dataset(data)


for data in ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Flickr', 'Yelp']:
    print('fetching '+ data)
    get_dataset(data, cuda=True)
