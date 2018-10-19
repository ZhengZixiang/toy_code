package me.zhengzx.recom.offline;

import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.precompute.MultithreadedBatchItemSimilarities;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;

public final class SimilarityTablesGenerator {
	
	private SimilarityTablesGenerator() {}

	public static void main(String[] args) throws IOException, TasteException, InterruptedException {
		DataModel dataModel = new GroupLensDataModel();
		UserItemSimilarityTableRedisWriter writer = new UserItemSimilarityTableRedisWriter(dataModel);
		writer.storeToRedis();
		
		ItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel, new LogLikelihoodSimilarity(dataModel));
		BatchItemSimilarities batch = new MultithreadedBatchItemSimilarities(recommender, 5);
		
		int numSimilarites = batch.computeItemSimilarities(Runtime.getRuntime().availableProcessors(), 1, new ItemSimilarityTableRedisWriter());
		
		System.out.println("Computed " + numSimilarites + " similarities for " + dataModel.getNumItems()
			+ " items and saved them to redis");
		
		writer.waitUtilDone();
	}
}
