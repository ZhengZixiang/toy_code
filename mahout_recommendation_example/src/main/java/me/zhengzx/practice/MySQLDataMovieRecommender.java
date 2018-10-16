package me.zhengzx.practice;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.JDBCDataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.mysql.jdbc.jdbc2.optional.MysqlDataSource;

public class MySQLDataMovieRecommender {

	public static void main(String[] args) throws TasteException, IOException {
		File resultFile = new File("/tmp", "mysqlrec.txt");
		
		//MySQL Connection
		MysqlDataSource dataSource = new MysqlDataSource();
		dataSource.setUseSSL(true);
		dataSource.setDatabaseName("mahout");
		dataSource.setServerName("localhost");
		dataSource.setUser("root");
		dataSource.setPassword("root");
		dataSource.setAutoReconnect(true);
        dataSource.setFailOverReadOnly(false);
		
		JDBCDataModel jdbcModel = new MySQLJDBCDataModel(dataSource, "taste_preferences", "user_id", "item_id", "preference", null);
		DataModel dataModel = jdbcModel;
		
		//Recommendation
		UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, dataModel);
		UserBasedRecommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
		
		
		try(PrintWriter writer = new PrintWriter(resultFile)) {
			for (int userID = 1; userID <= dataModel.getNumUsers(); userID++) {
				System.out.println("----------------------------------------------------");
				List<RecommendedItem> items = recommender.recommend(userID, 3);
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
