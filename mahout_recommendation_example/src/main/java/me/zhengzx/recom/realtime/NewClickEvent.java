package me.zhengzx.recom.realtime;

public class NewClickEvent {
	private long userID;
	private long itemID;
	
	public NewClickEvent() {
		this.userID = -1L;
		this.itemID = -1L;
	}
	
	public NewClickEvent(long userID, long itemID) {
		this.userID = userID;
		this.itemID = itemID;
	}
	
	public long getUserID() {
		return userID;
	}
	
	public void setUserID(long userID) {
		this.userID = userID;
	}
	
	public long getItemID() {
		return itemID;
	}
	
	public void setItemID(long itemID) {
		this.itemID = itemID;
	}
}
