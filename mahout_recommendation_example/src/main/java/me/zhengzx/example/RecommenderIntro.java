package me.zhengzx.example;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;;

public class RecommenderIntro {
	
	public static void main(String[] args) throws IOException, TasteException {
		DataModel dataModel = new FileDataModel(new File("./ua.base"));
		UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);
		Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
		
		List<RecommendedItem> list = recommender.recommend(1, 20);  //第二个参数大于K近邻时只会输出K个值
		
		for(RecommendedItem item : list) {
			System.out.println(item);
		}
	}
}
