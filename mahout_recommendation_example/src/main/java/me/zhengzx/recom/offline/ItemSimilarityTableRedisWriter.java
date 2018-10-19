package me.zhengzx.recom.offline;

import java.io.IOException;

import org.apache.mahout.cf.taste.similarity.precompute.SimilarItem;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItems;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItemsWriter;

import com.alibaba.fastjson.JSON;

import me.zhengzx.recom.common.ItemSimilarity;
import me.zhengzx.recom.common.RedisUtil;
import redis.clients.jedis.Jedis;

public class ItemSimilarityTableRedisWriter implements SimilarItemsWriter {

	private long itemCounter = 0;
	private Jedis jedis = null;
	
	@Override
	public void close() throws IOException {
		jedis.close();
	}

	@Override
	public void open() throws IOException {
		jedis = RedisUtil.getJedis();
	}

	@Override
	public void add(SimilarItems similarItems) throws IOException {
		ItemSimilarity[] values = new ItemSimilarity[similarItems.numSimilarItems()];
		int counter = 0;
		for (SimilarItem item : similarItems.getSimilarItems()) {
			values[counter] = new ItemSimilarity(item.getItemID(), item.getSimilarity());
			counter++;
		}
		String key = "II:" + similarItems.getItemID();
		String value = JSON.toJSONString(values);
		jedis.set(key, value);
		itemCounter++;
		if(itemCounter % 100 == 0) {
			System.out.println("Store " + key + " to redis, total: " + itemCounter);
		}
	}

}
