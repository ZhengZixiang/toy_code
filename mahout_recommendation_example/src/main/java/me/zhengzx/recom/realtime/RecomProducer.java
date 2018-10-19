package me.zhengzx.recom.realtime;


import java.util.Properties;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
//import org.apache.kafka.clients.producer.ProducerConfig
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.log4j.Logger;

import com.alibaba.fastjson.JSON;

import me.zhengzx.recom.common.Constants;

public class RecomProducer implements Runnable {
	private static final Logger LOGGER = Logger.getLogger(RecomProducer.class);
	
	private final String topic;
	
	public RecomProducer(String topic) {
		this.topic = topic;
	}
	
	static NewClickEvent[] events = new NewClickEvent[] {
			new NewClickEvent(1000000L, 123L),
			new NewClickEvent(1000001L, 400L),
			new NewClickEvent(1000002L, 500L),
			new NewClickEvent(1000003L, 278L),
			new NewClickEvent(1000004L, 681L),
	};

	public void run() {
		Properties properties = new Properties();
		properties.put("bootstrap.servers", Constants.KAFKA_ADDR);
		properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		properties.put("acks", "all");
		properties.put("retries", 1);
		Producer<Integer, String> producer = null;
		
		System.out.println("Producing messages");
		try {
			producer = new KafkaProducer<Integer, String>(properties);
			for (NewClickEvent event : events) {
				String eventAsStr = JSON.toJSONString(event);
				producer.send(new ProducerRecord<Integer, String>(topic, eventAsStr));
				System.out.println("Sending messages: " + eventAsStr);
			}
			System.out.println("Done sending messages");
		} catch (Exception e) {
			LOGGER.fatal("Error while producing messages", e);
			LOGGER.trace(null, e);
			System.err.println("Error while producing messages: " + e);
		} finally {
			if (producer != null) {
				producer.close();
			}
		}
	}
	
	public static void main(String[] args) throws Exception {
		new Thread(new RecomProducer(Constants.KAFKA_TOPICS)).start();
	}

}
