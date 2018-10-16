package me.zhengzx.movielens;

import java.io.File;

import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.precompute.FileSimilarItemsWriter;
import org.apache.mahout.cf.taste.impl.similarity.precompute.MultithreadedBatchItemSimilarities;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItemsWriter;

public class MovieLensBatchItemSimilarities {
	
	//argument: ./ratings.dat
	public static void main(String[] args) throws Exception {
		if(args.length != 1) {
			System.err.println("Needs MovieLens 1M dataset as argument!");
			System.exit(-1);
		}
		
		File resultFile = new File(System.getProperty("java.io.tmpdir"), "similarities.csv");
		
		DataModel dataModel = new MovieLensDataModel(new File(args[0]));
		ItemSimilarity similarity = new LogLikelihoodSimilarity(dataModel);
		ItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel, similarity);
		BatchItemSimilarities similarities = new MultithreadedBatchItemSimilarities(recommender, 5); //batchSize
		
		SimilarItemsWriter writer = new FileSimilarItemsWriter(resultFile);
		int numSimilarities = similarities.computeItemSimilarities(Runtime.getRuntime().availableProcessors(),
				1, writer);
		
		System.out.println("Computed " + numSimilarities + " for " + dataModel.getNumItems() +
				" items and saved them to  " + resultFile.getAbsolutePath());
	}

}
