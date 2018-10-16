package me.zhengzx.bookcrossing;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.model.DataModel;

public class BXRecommenderEvaluator {
	
	//argument: ./BX-Book-Ratings.csv
	public static void main(String[] args) throws TasteException, IOException {
		DataModel dataModel = new BXDataModel(new File(args[0]), false);
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		double score = evaluator.evaluate(new BXRecommenderBuilder(), new BXDataModelBuilder(), dataModel, 0.9, 0.3);
		
		System.out.println("MAE score: " + score);
	}
	
}
