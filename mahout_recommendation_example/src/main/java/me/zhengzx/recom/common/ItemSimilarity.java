package me.zhengzx.recom.common;

public class ItemSimilarity implements Comparable<ItemSimilarity>{
	private long itemID;
	private Double similarity;

	public ItemSimilarity() {
		this.itemID = -1;
		this.similarity = 0d;
	}
	
	public ItemSimilarity(long itemID, double similarity) {
		this.itemID = itemID;
		this.similarity = similarity;
	}
	
	public long getItemID() {
		return itemID;
	}

	public void setItemID(long itemID) {
		this.itemID = itemID;
	}

	public Double getSimilarity() {
		return similarity;
	}

	public void setSimilarity(Double similarity) {
		this.similarity = similarity;
	}

	@Override
	public int compareTo(ItemSimilarity obj) {
		if(this.similarity > obj.similarity) {
			return 1;
		} else if(this.similarity < obj.similarity) {
			return -1;
		} else {
			return 0;
		}
	}

	@Override
	public boolean equals(Object obj) {
		if(!(obj instanceof ItemSimilarity)) {
			return false;
		}
		if(obj == this) {
			return true;
		}
		return this.itemID == ((ItemSimilarity) obj).itemID &&
				this.similarity == ((ItemSimilarity) obj).similarity;
	}

	@Override
	public int hashCode() {
		return (int)(itemID+similarity);
	}

	@Override
	public String toString() {
		return "item id: " + itemID + ", similarity: " + similarity;
	}

}
