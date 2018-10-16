package me.zhengzx.movielens;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class MovieLensUserRecommender {
	
	//argument: ./ratings.dat
	public static void main(String[] args) throws Exception {
		if(args.length != 1) {
			System.err.println("Needs MovieLens 1M dataset as argument!");
			System.exit(-1);
		}
		
		File resultFile = new File(System.getProperty("java.io.tmpdir"), "userrecommend.csv");
		
		DataModel dataModel = new MovieLensDataModel(new File(args[0]));
		UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);
		
		Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
		Recommender caching = new CachingRecommender(recommender);
		
		//Evaluate
		RMSRecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
		RecommenderBuilder builder = new RecommenderBuilder() {
			@Override
			public Recommender buildRecommender(DataModel dataModel) throws TasteException {
				UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);
				Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
				return recommender;
			}
		};
		double score = evaluator.evaluate(builder, null, dataModel, 0.9, 0.5);
		System.out.println("RMS score is " + score);
		
		try(PrintWriter writer = new PrintWriter(resultFile)) {
			for (int userID = 1; userID <= dataModel.getNumUsers(); userID++) {
				List<RecommendedItem> items = caching.recommend(userID, 5);
				String line = userID + " : ";
				for (RecommendedItem item : items) {
					line += item.getItemID()+":"+item.getValue()+",";
				}
				if (line.endsWith(",")) {
					line.substring(0, line.length()-2);
				}
				writer.write(line + '\n');
			}
		} catch (IOException e) {
			resultFile.delete();
			throw e;
		}
		System.out.println("Recommended for " + dataModel.getNumUsers() + " users and saved them to " + resultFile.getAbsolutePath());
	}
	
}
