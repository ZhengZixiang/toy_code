package me.zhengzx.recom.offline;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.URL;
import java.util.regex.Pattern;

import org.apache.commons.io.Charsets;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

import com.google.common.io.Files;
import com.google.common.io.InputSupplier;
import com.google.common.io.Resources;

public class GroupLensDataModel extends FileDataModel {

	private static final long serialVersionUID = 1L;
	private static final String COLON_DELIMITER = "::";
	private static final Pattern COLON_DELIMITER_PATTERN = Pattern.compile(COLON_DELIMITER);
			
	public GroupLensDataModel() throws IOException {
		this(readResourceToTempFile("./src/main/resources/ratings.dat"));
	}
	
	public GroupLensDataModel(File ratingsFile) throws IOException {
		super(convertGLFile(ratingsFile));
	}
	
	private static File convertGLFile(File originalFile) throws IOException {
		File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "ratings.txt");
		if (resultFile.exists()) {
			resultFile.delete();
		}
		try(Writer writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8)) {
			for (String line : new FileLineIterable(originalFile, false)) {
				int lastDelimiterStart = line.lastIndexOf(COLON_DELIMITER);
				if(lastDelimiterStart < 0) {
					throw new IOException("Unexpected input format on line: " + line);
				}
				String subLine = line.substring(0, lastDelimiterStart);
				String convertedLine = COLON_DELIMITER_PATTERN.matcher(subLine).replaceAll(",");
				writer.write(convertedLine);
				writer.write('\n');
			}
		} catch (IOException e){
			resultFile.delete();
			throw e;
		}
		return resultFile;
	}
	

	public static File readResourceToTempFile(String resourceName) throws IOException {
		InputSupplier<? extends InputStream> inputSupplier;
		try {
			URL resourceURL = Resources.getResource(GroupLensDataModel.class, resourceName);
			inputSupplier = Resources.newInputStreamSupplier(resourceURL);
		} catch (IllegalArgumentException e) {
			File resourceFile = new File("./" + resourceName);
			inputSupplier = Files.newInputStreamSupplier(resourceFile);
		}
		File tempFile = File.createTempFile("taste", null);
		tempFile.deleteOnExit();
		Files.copy(inputSupplier, tempFile);
		return tempFile;
	}

	@Override
	public String toString() {
		return "GroupLensDataModel";
	}
	
}
