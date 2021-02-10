pip3 install --user torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user pygsp
pip3 install --user scikit_learn
pip3 install --user torch-scatter==2.0.4+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-sparse==0.6.4+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-cluster==1.5.4+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-spline-conv==1.2.0+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --user torch-geometric
python3 setup.py install --user
python3 citation/fetch_data.py
