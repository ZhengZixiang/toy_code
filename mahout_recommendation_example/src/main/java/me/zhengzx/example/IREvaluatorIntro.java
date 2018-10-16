package me.zhengzx.example;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

public class IREvaluatorIntro {
	public static void main(String[] args) throws IOException, TasteException {
		RandomUtils.useTestSeed();
		
		DataModel dataModel = new FileDataModel(new File("./ua.base"));
		RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
		RecommenderBuilder builder = new RecommenderBuilder() {
			@Override
			public Recommender buildRecommender(DataModel dataModel) throws TasteException {
				UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, dataModel);
				return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
			}
		};
		
		IRStatistics stats = evaluator.evaluate(builder, null, dataModel, null, 5, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
		System.out.println(stats);
		System.out.println(stats.getPrecision());
		System.out.println(stats.getRecall());
		System.out.println(stats.getF1Measure());
	}
}
