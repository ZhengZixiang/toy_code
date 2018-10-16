package me.zhengzx.example;

import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class CreatePreferenceArray {

	public static void main(String[] args) {
		PreferenceArray user1Pref = new GenericUserPreferenceArray(2);
		user1Pref.setUserID(0, 1L);
		user1Pref.setItemID(0, 101L);
		user1Pref.setValue(0, 3.0f);
		user1Pref.setItemID(1, 102L);
		user1Pref.setValue(1, 4.0f);
		Preference pref = user1Pref.get(1);
		System.out.println(pref);
		System.out.println(user1Pref);
	}

}
