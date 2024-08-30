DIR=$(pwd)

cd $DIR/mycpp/ && rm -rf build && mkdir build && cd build && cmake .. && make -j11
# cd /kaolin && rm -rf build *egg* && pip install -e .
# cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
