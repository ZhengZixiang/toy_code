package me.zhengzx.recom.webservice;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

import com.alibaba.fastjson.JSON;

import me.zhengzx.recom.common.ItemSimilarity;
import me.zhengzx.recom.common.RedisUtil;
import redis.clients.jedis.Jedis;

@Path("/item/recom")
public class ItemBasedRecResult {
	Jedis jedis = null;
	
	public ItemBasedRecResult() {
		jedis = RedisUtil.getJedis();
	}

	@GET
	@Path("/{userID}")
	@Produces(MediaType.APPLICATION_JSON)
	public RecommendedItems getRecItems(@PathParam("userID") String userID) {
		RecommendedItems recItems = new RecommendedItems();
		
		//Stage 1: get user's items
		String key = String.format("UI:%s", userID);
		String value = jedis.get(key);
		if (value == null || value.length() <= 0) {
			return recItems;
		}
		List<Long> userItems = JSON.parseArray(value, Long.class);
		Set<Long> userItemSet = new TreeSet<Long>(userItems);
		
		//Stage 2: get similar items to the user's items
		List<String> userItemStrs = new ArrayList<String>();
		for (Long item : userItems) {
			userItemStrs.add("II:" + item);
		}
		
		List<String> similarItems = jedis.mget(userItemStrs.toArray(new String[userItemStrs.size()]));
		Set<ItemSimilarity> similarItemsSet = new TreeSet<ItemSimilarity>();
		for (String item : similarItems) {
			List<ItemSimilarity> result = JSON.parseArray(item, ItemSimilarity.class);
			similarItemsSet.addAll(result);
		}
		
		List<Long> recommendedItemIDs = new ArrayList<Long>();
		for (ItemSimilarity is : similarItemsSet) {
			if(!userItemSet.contains(is.getItemID())) {
				recommendedItemIDs.add(is.getItemID());
			}
			if(recommendedItemIDs.size() >= 10) {
				break;
			}
		}
		//recItems.setItems(recommendedItemIDs.toArray(new Long[recommendedItemIDs.size()]));
		recItems.setItems(recommendedItemIDs.toArray(new Long[0]));
		return recItems;
		
	}
}
