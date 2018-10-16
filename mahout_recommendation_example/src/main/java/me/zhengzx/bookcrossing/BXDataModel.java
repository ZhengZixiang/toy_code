package me.zhengzx.bookcrossing;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.regex.Pattern;

import org.apache.commons.io.Charsets;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

public class BXDataModel extends FileDataModel {

	private static final long serialVersionUID = 1L;
	
	private static Pattern NON_DIGITAL_SEMICOLON_DELIMITER = Pattern.compile("[^0-9;]");

	public BXDataModel(File originalFile, Boolean ignoreRatings) throws IOException {
		super(convertFile(originalFile, ignoreRatings));
	}

	private static File convertFile(File originalFile, Boolean ignoreRatings) throws IOException {
		File resultFile = new File(System.getProperty("java.io.tmpdir"), "bookcrossing.csv");
		if (resultFile.exists()) {
			resultFile.delete();
		}
		
		try (Writer writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8)) {
			for (String line : new FileLineIterable(originalFile, true)) {
				if (line.endsWith("\"0\"")) {
					continue;
				}
				String convertedLine = NON_DIGITAL_SEMICOLON_DELIMITER.matcher(line).replaceAll("").replace(';', ',');
				
				if (convertedLine.contains(",,")) {
					continue;
				}
				if (ignoreRatings) {
					convertedLine = convertedLine.substring(0,  convertedLine.lastIndexOf(','));
				}
				
				writer.write(convertedLine);
				writer.write('\n');
			}
		} catch (IOException e){
			resultFile.delete();
			throw e;
		}
		return resultFile;
	}
	
}
