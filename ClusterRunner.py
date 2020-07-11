from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score


class ClusterRunner():
    def get_clus_ground_truth(self,ground_truth_list, cluster_labels):
        """Takes a ground truth list, which is the labels as to whether or not
        the sentence has a polymer. The cluster_labels refers to the labels the clustering
        algorithm assigned to the observation."""
        
        ground_truths = {}
        for ix in range(len(ground_truth_list)):
            
            cluster = cluster_labels[ix]
            ground_truth = ground_truth_list[ix]
            
            if cluster not in ground_truths:
                ground_truths[cluster] = {1: 0, 0: 0}

            ground_truths[cluster][ground_truth] += 1


        return ground_truths

    def get_single_clus_homogeneity(self,value_counts_dict,min_size,threshold):
        """Gets homogeneity of each single cluster and returns them in a tuppled list for clusters with size
        > min_size and with homogeneity > threshold """
        matching_clusters = []

        for cluster in value_counts_dict:

            cluster_counts = value_counts_dict[cluster]
            yes_count = cluster_counts[1]
            no_count = cluster_counts[0]

            total = yes_count + no_count
            if total > min_size:
                numerator = no_count if no_count > yes_count else yes_count
                label = "no" if no_count > yes_count else "yes"
                homogeneity = numerator / total
                if homogeneity > threshold:
                    matching_clusters.append((cluster, homogeneity, label, numerator))

        return matching_clusters
    
class KMeansRunner(ClusterRunner):
    
    def __init_dict(self,n_grams,n_clusters,k_obj):
        return {n_gram: {k: k_obj for k in n_clusters} for n_gram in n_grams}
    
    def run(self,
    n_grams,
    n_clusters,
    sentences,
    ground_truth,
    vectorizer_class,
    min_size=500,
    min_threshold=0.95):
    
        all_cluster_h_scores = self.__init_dict(n_grams,n_clusters,None) #homogeneity
        all_cluster_s_scores = self.__init_dict(n_grams,n_clusters,None) #silhouette    
        single_cluster_h_scores = self.__init_dict(n_grams,n_clusters,[])
        models = self.__init_dict(n_grams,n_clusters,[]) #stores the model obj for each run

        for n_gram in n_grams:
            print(n_gram)
            vectorizer = vectorizer_class(ngram_range=(n_gram))
            X = vectorizer.fit_transform(sentences)

            for k in n_clusters:
                print(k)

                kmeans = KMeans(n_clusters=k).fit(X)
                all_cluster_h_scores[n_gram][k] = homogeneity_score(ground_truth,kmeans.labels_)                
                all_cluster_s_scores[n_gram][k] = silhouette_score(X,kmeans.labels_)
                models[n_gram][k] = kmeans

                homogeneities = self.get_clus_ground_truth(ground_truth,kmeans.labels_)
                single_cluster_h_scores[n_gram][k] = self.get_single_clus_homogeneity(homogeneities, min_size=min_size,threshold=min_threshold)

        return {"homogeneity_all":all_cluster_h_scores,
        "homogeneity_single":single_cluster_h_scores,
        "models":models,
        "silhouette":all_cluster_s_scores}



class DBScanRunner(ClusterRunner):
    def __init_dict(self,n_grams,obj):
        return {n_gram: obj for n_gram in n_grams}
    
    def run(self,
    n_grams,
    sentences,
    ground_truth,
    vectorizer_class,
    dbscan_kwargs,
    min_size=500,
    min_threshold=0.95):
    
        all_cluster_h_scores = self.__init_dict(n_grams,None) #homogeneity
        all_cluster_s_scores = self.__init_dict(n_grams,None) #silhouette    
        single_cluster_h_scores = self.__init_dict(n_grams,[])
        models = self.__init_dict(n_grams,None)

        for n_gram in n_grams:
            print(n_gram)
            vectorizer = vectorizer_class(ngram_range=(n_gram))
            X = vectorizer.fit_transform(sentences)

            cluster = DBSCAN(**dbscan_kwargs).fit(X)
            all_cluster_h_scores[n_gram] = homogeneity_score(ground_truth,cluster.labels_)                
            all_cluster_s_scores[n_gram] = silhouette_score(X,cluster.labels_)
            models[n_gram]= cluster

            homogeneities = self.get_clus_ground_truth(ground_truth,cluster.labels_)
            single_cluster_h_scores[n_gram] = self.get_single_clus_homogeneity(homogeneities, min_size=min_size,threshold=min_threshold)

        return {"homogeneity_all":all_cluster_h_scores,
        "homogeneity_single":single_cluster_h_scores,
        "models": models,
        "silhouette":all_cluster_s_scores}


            