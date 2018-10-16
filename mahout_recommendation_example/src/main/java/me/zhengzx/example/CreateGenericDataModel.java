package me.zhengzx.example;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class CreateGenericDataModel {

	public static void main(String[] args) {
		FastByIDMap<PreferenceArray> preferences = new FastByIDMap<PreferenceArray>();
		PreferenceArray user1Pref = new GenericUserPreferenceArray(2);
		user1Pref.setUserID(0, 1L);
		user1Pref.setItemID(0, 101L);
		user1Pref.setValue(0, 3.0f);
		user1Pref.setItemID(1, 102L);
		user1Pref.setValue(1, 4.0f);
		preferences.put(1L, user1Pref);
		
		PreferenceArray user2Pref = new GenericUserPreferenceArray(2);
		user1Pref.setUserID(0, 2L);
		user1Pref.setItemID(0, 101L);
		user1Pref.setValue(0, 3.0f);
		user1Pref.setItemID(1, 102L);
		user1Pref.setValue(1, 4.0f);
		preferences.put(2L, user2Pref);
		
		DataModel dataModel = new GenericDataModel(preferences);
		System.out.println(dataModel);
	}

}
