# import numpy as np
# import gensim
# from gensim.models import Word2Vec
# from matplotlib import pyplot
# import sklearn
# from sklearn.decomposition import PCA
#
#
# def loadGloveModel(gloveFile):
#     print("Loading Glove Model")
#     f = open(gloveFile,'r', encoding='utf8')
#     model = {}
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#     print("Done.",len(model)," words loaded!")
#     return model
#
# # model = loadGloveModel("glove/glove.6B.50d.txt")
# sentences = "SSB might be an acronym or abbreviation for:\n\n* Super Smash Bros. series** Super Smash Bros., a game in the series* Sacramento Sustainable Business, environmentally-friendly business recognition program operated by the Business Environmental Resource Center. * Salomon Smith Barney, a former investment firm* Sathya Sai Baba* Sauder School of Business, at the University of British Columbia* Same-sex blessings: the Blessing of same-sex unions in Christian churches* Schulich School of Business, in York University* Services Selection Board, for selecting cadets in Indian military* Ship Submersible Ballistic, submarine* Single-sideband modulation, in radio technology* Single-strand binding protein* Single-stranded DNA Break* Site-specific browser* Sjogren syndrome antigen B, a human gene* Societas Sanctae Birgittae, the Society of Saint Bridget* Society of the Sisters of Bethany, an Anglican Religious order* Special Service Battalion, a South African military unit* Spontaneous symmetry breaking, in physics* Star Spangled Banner, national anthem of the United States* State Street Bank* Statistisk Sentralbyrå, Norwegian statistics bureau * Strategic Support Branch* Stuttgarter Straßenbahnen AG, public transport operator in Stuttgart, Germany* Sunshine Skyway Bridge* Sustainable South Bronx, environmental justice organization* Secret Service Bureau of the United Kingdom, now called the Secret Intelligence Service{{disambig}}Category:AcronymsCategory:Abbreviations\n\nde:SSBes:SSBfr:SSBit:SSBlt:SSBja:SSBno:SSB (andre betydninger)pl:SSBzh:SSB"
# model = Word2Vec([i.lower().split(' ') for i in sentences.split(".")], min_count=1, size=50)
#
# X = model[model.wv.vocab]
#
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
#
# print(model['SSB'])
