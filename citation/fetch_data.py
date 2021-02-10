from citation import get_dataset


for data in ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Flickr', 'Yelp']:
    print('fetching '+ data)
    get_dataset(data, cuda=True)

