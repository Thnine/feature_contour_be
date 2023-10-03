from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json

from sklearn.manifold import TSNE

from DimReader.DimReader import runDimReader
from EMD.emd import getEMD


app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route('/requestContour',methods=['POST'])
def requestContour():
    reqParams = json.loads(request.get_data())

    allMatrix = reqParams['allMatrix']
    usedMatrix = reqParams['usedMatrix']
    perturbationIndex = reqParams['perturbationIndex']
    
    tsne_result = TSNE(n_components=2,init='pca',random_state=1).fit_transform(usedMatrix)

    

    if perturbationIndex == -1: #如果不需要计算等高线
        result = {
            'points': tsne_result.tolist()
        }        
    else: #如果需要计算等高线
        result = runDimReader(allMatrix,tsne_result,perturbationIndex,[])



    return json.dumps(result)

@app.route('/requestFeatureSort',methods=['POST'])
def requestFeatureSort():
    reqParams = json.loads(request.get_data())

    features = reqParams['features']
    labels = reqParams['labels']
    dataH = reqParams['dataH']
    ticks = reqParams['ticks']

    ## 按照emd排序
    emd_scores={}
    for f in features:
        score = 0
        tickByFeature = np.array([ticks[f][:-1]]).T
        for i1 in range(0,len(labels)):
            for i2 in range(i1 + 1,len(labels)):
                e1 = labels[i1]
                e2 = labels[i2]
                # for i in range(0,10):
                #   score += abs(dataH[e1][f][i] - dataH[e2][f][i])
                score += getEMD((tickByFeature,np.array(dataH[e1][f])), (tickByFeature,np.array(dataH[e2][f])))
        emd_scores[f] = score

    sorted_feature = sorted(emd_scores.items(),key=lambda x:x[1])
    sorted_feature.reverse()

    return sorted_feature


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5051)
