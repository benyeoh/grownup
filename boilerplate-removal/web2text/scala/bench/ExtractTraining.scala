package bench

import ch.ethz.dalab.web2text.utilities.Util
import ch.ethz.dalab.web2text.cleaneval.CleanEval
import ch.ethz.dalab.web2text.cdom.{CDOM,DOM}
import org.jsoup.Jsoup
import ch.ethz.dalab.web2text.features.extractor._
import ch.ethz.dalab.web2text.classification.{PerformanceStatistics}
import ch.ethz.dalab.web2text.features.{BlockFeatureExtractor,FeatureExtractor}
import ch.ethz.dalab.web2text.alignment.Alignment
import ch.ethz.dalab.web2text.features.PageFeatures
import com.mongodb.casbah.Imports._
import java.io._
import ch.ethz.dalab.web2text.output.CsvDatasetWriter
//import ch.ethz.dalab.web2text.utilities.Warc
//import ch.ethz.dalab.web2text.output.CleanTextOutput
import scala.util.{Try,Success,Failure}
import breeze.linalg.{DenseMatrix,csvwrite,DenseVector}
import breeze.io.CSVWriter


object ExtractTraining {

    def main(args: Array[String]): Unit = {

        println(s"Finding html/clean pairs in (${args(0)}, ${args(1)})\n")
        val filteredPairs = Dataset.findHTMLCleanPair(args(0), args(1))
        println(s"Found ${filteredPairs.length} pairs")

        println("Extracting features / labels / texts")
        val extracted = Dataset.extractAll(filteredPairs).filter(d => d._1 != null).map {
            data => (data._1, data._2)
        }

        println(s"Writing features to ${args(2)}")
        //CsvDatasetWriter.write(extracted, args(2))
        
        val dir = new File(args(2))
        dir.mkdirs()

        val blockLabels = extracted.head._1.blockFeatureLabels
        val edgeLabels  = extracted.head._1.edgeFeatureLabels

        if (blockLabels.length > 0) {
            val rows = for ( ((features, labels), i) <- extracted.zipWithIndex;
                            (l,j) <- labels.zipWithIndex;
                            row = features.blockFeatures(::,j) )
                        yield (Array(i.toDouble, l.toDouble) ++ row.toArray)

            val outFile = new FileWriter(new File(args(2), "block_features.csv"))
            rows.foreach(row => {
                val mat = new DenseMatrix(row.length, 1, row).t
                val tab = IndexedSeq.tabulate(mat.rows, mat.cols)(mat(_, _).toString)
                CSVWriter.write(outFile, tab, ',', '\u0000', '\\')
            })             
            outFile.close()

            // csvwrite(
            //     new File(dir, "block_features.csv"),
            //     new DenseMatrix(rows.head.length, rows.length, rows.flatten.toArray).t
            // )
        }

        if (edgeLabels.length > 0) {
            val rows = for ( ((features, labels), i) <- extracted.zipWithIndex;
                            ((l1,l2),j) <- (labels zip labels.tail).zipWithIndex;
                            row = features.edgeFeatures(::,j) )
                        yield (Array(i.toDouble, (l1*10+l2).toDouble) ++ row.toArray)

            val outFile = new FileWriter(new File(args(2), "edge_features.csv"))
            rows.foreach(row => {
                val mat = new DenseMatrix(row.length, 1, row).t
                val tab = IndexedSeq.tabulate(mat.rows, mat.cols)(mat(_, _).toString)
                CSVWriter.write(outFile, tab, ',', '\u0000', '\\')
            })
            outFile.close()

            // csvwrite(
            //     new File(dir, "edge_features.csv"),
            //     new DenseMatrix(rows.head.length, rows.length, rows.flatten.toArray).t
            // )
        }

        println("Done!\n")
    }
}