package me.zhengzx.movielens;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.regex.Pattern;

import org.apache.commons.io.Charsets;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

public class MovieLensDataModel extends FileDataModel {

	private static final long serialVersionUID = 1L;
	
	private static String COLON_DELIMITER = "::";
	private static Pattern COLON_DELIMITER_PATTERN = Pattern.compile(COLON_DELIMITER);

	public MovieLensDataModel(File originalFile) throws IOException {
		super(convertFile(originalFile));
	}

	private static File convertFile(File originalFile) throws IOException {
		File resultFile = new File(System.getProperty("java.io.tmpdir"), "rating.csv");
		if (resultFile.exists()) {
			resultFile.delete();
		}
		
		//Writer writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8);
		//下面的方式就不需要close writer
		try(Writer writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8)) {
			for (String line : new FileLineIterable(originalFile, false)) {
				int lastIndex = line.lastIndexOf(COLON_DELIMITER);
				if(lastIndex < 0) {
					throw new IOException("Invalid data!");
				}
				String subLine = line.substring(0, lastIndex);
				String convertedSubLine = COLON_DELIMITER_PATTERN.matcher(subLine).replaceAll(",");
				writer.write(convertedSubLine);
				writer.write('\n');
			}
		} catch (IOException e){
			resultFile.delete();
			throw e;
		}
		return resultFile;
	}
	
}
