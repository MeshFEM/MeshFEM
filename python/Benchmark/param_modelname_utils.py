'''
Python Helper Functions for model name conversion in paper

Author:  Xinzhuo (johnson) Hu
Created: 10/14/2025  21:53:35
'''


def getModelNameDict():
    '''
    get a model name dict for 24 models in param benchmark
    '''
    modelNameDict = {}
    modelNameDict['cow2Disc'] = 'cow'
    modelNameDict['bumpy_sphereDisc'] = 'bumpy-sphere'
    modelNameDict['denteDisc'] = 'dente'
    modelNameDict['armadilloDisc'] = 'armadillo'
    modelNameDict['davidDisc'] = 'david'
    modelNameDict['bladeDisc'] = 'blade'
    modelNameDict['hand'] = 'hand'
    modelNameDict['gargoyle_cut'] = 'gargoyle'
    modelNameDict['vase_lion'] = 'vase-lion'
    modelNameDict['bimba100KDisc'] = 'bimba'
    modelNameDict['busteDisc'] = 'buste'
    modelNameDict['armchairDisc'] = 'armchair'
    modelNameDict['deformed_armadilloDisc'] = 'deformed-armadillo'
    modelNameDict['camille_hand100KDisc'] = 'camille-hand'
    modelNameDict['bunnyBotschDisc'] = 'bunny2'
    modelNameDict['Superman_cut2'] = 'superman2'
    modelNameDict['Superman_cut3'] = 'superman3'
    modelNameDict['Superman_cut1'] = 'superman1'
    modelNameDict['bear_cut'] = 'bear'
    modelNameDict['dragonHead2'] = 'dragon-head'
    modelNameDict['eros'] = 'eros'
    modelNameDict['buddha_cut'] = 'buddha'
    modelNameDict['Lucy_3cuts'] = 'lucy'
    modelNameDict['chinese_dragon'] = 'chinese-dragon'

    return modelNameDict


if __name__ == "__main__":
    print("Hello World")