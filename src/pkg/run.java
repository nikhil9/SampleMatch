package pkg;
import com.googlecode.javacv.cpp.opencv_features2d.CvSURFPoint;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import pkg.SampleMatch.*;

public class run{
    public static void main(String[] args) throws Exception {

        IplImage sample = cvLoadImage("sample.png", CV_LOAD_IMAGE_GRAYSCALE);
        IplImage image  = cvLoadImage("test1.png",  CV_LOAD_IMAGE_GRAYSCALE);       

        IplImage SampleColor = IplImage.create(sample.width(), sample.height(), 8, 3);
        cvCvtColor(sample, SampleColor, CV_GRAY2BGR);

        IplImage correspond = IplImage.create(image.width(), sample.height()+ image.height(), 8, 1);
        cvSetImageROI(correspond, cvRect(0, 0, sample.width(), sample.height()));
        cvCopy(sample, correspond);
        cvSetImageROI(correspond, cvRect(0, sample.height(), correspond.width(), correspond.height()));
        cvCopy(image, correspond);
        cvResetImageROI(correspond);

        sampleSettings settings = new sampleSettings();
        settings.sampleImage = sample;
        settings.useFLANN = true;
        settings.ransacReprojThresh = 5;
        SampleMatch finder = new SampleMatch(settings);

        long start = System.currentTimeMillis();
        double[] dst_corners = finder.find(image);
        System.out.println("Finding time = " + (System.currentTimeMillis() - start) + " ms");

        if (dst_corners !=  null) {
            for (int i = 0; i < 4; i++) {
                int j = (i+1)%4;
                int x1 = (int)Math.round(dst_corners[2*i    ]);
                int y1 = (int)Math.round(dst_corners[2*i + 1]);
                int x2 = (int)Math.round(dst_corners[2*j    ]);
                int y2 = (int)Math.round(dst_corners[2*j + 1]);
                cvLine(correspond, cvPoint(x1, y1 + sample.height()),
                        cvPoint(x2, y2 + sample.height()),
                        CvScalar.WHITE, 1, 8, 0);
            }
        }

        for (int i = 0; i < finder.ptpairs.size(); i += 2) {
            CvPoint2D32f pt1 = finder.objectKeypoints[finder.ptpairs.get(i)].pt();
            CvPoint2D32f pt2 = finder.imageKeypoints[finder.ptpairs.get(i+1)].pt();
            cvLine(correspond, cvPointFrom32f(pt1),
                    cvPoint(Math.round(pt2.x()), Math.round(pt2.y()+sample.height())),
                    CvScalar.WHITE, 1, 8, 0);
        }       
        
        cvNamedWindow("Sample");
        cvNamedWindow("Sample Correspond");
        cvShowImage("Sample Correspond", correspond);
        
        for (int i = 0; i < finder.objectKeypoints.length; i++ ) {
            CvSURFPoint r = finder.objectKeypoints[i];
            CvPoint center = cvPointFrom32f(r.pt());
            int radius = Math.round(r.size()*1.2f/9*2);
            cvCircle(SampleColor, center, radius, CvScalar.RED, 1, 8, 0);
        }
        
        cvShowImage("Sample", SampleColor);
        cvWaitKey(0);
        
    }
}
