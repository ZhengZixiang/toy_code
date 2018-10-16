package me.zhengzx.example;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

public class EvaluatorIntro {
	
	public static void main(String[] args) throws IOException, TasteException {
		RandomUtils.useTestSeed();
		
		DataModel dataModel = new FileDataModel(new File("./ua.base"));
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderEvaluator evaluator2 = new RMSRecommenderEvaluator();
		RecommenderBuilder builder = new RecommenderBuilder() {
			
			@Override
			public Recommender buildRecommender(DataModel dataModel) throws TasteException {
				UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);
				return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
			}
		};
		
		//trainingPercentage 代表一个用户取多少比例来训练
		//evaluationPercentage 代表取多少比例用户来测试
		double score = evaluator.evaluate(builder, null, dataModel, 0.7, 1);
		double rmse = evaluator2.evaluate(builder, null, dataModel, 0.7, 1);
		
		System.out.println(score + " " + rmse);
	}
	
}
