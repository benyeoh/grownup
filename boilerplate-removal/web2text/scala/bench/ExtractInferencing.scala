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
import ch.ethz.dalab.web2text.utilities.Warc
import ch.ethz.dalab.web2text.output.CleanTextOutput
import scala.util.{Try,Success,Failure}
import breeze.linalg.csvwrite


object ExtractInferencing {

    def main(args: Array[String]): Unit = {

        println(s"Finding html/clean pairs in (${args(0)}, ${args(1)})\n")
        val filteredPairs = Dataset.findHTMLCleanPair(args(0), args(1))
        println(s"Found ${filteredPairs.length} pairs")

        println("Extracting features / texts")
        val extracted = Dataset.extractFeatsAndText(filteredPairs map { p => p._1 })

        (filteredPairs zip extracted).foreach{ case (pair, extract) =>
            if (extract._1 != null && extract._1.nBlocks >= 2) {
                val name = pair._1.split("/").last.split(".html")(0)
                val dirPath = new File(args(2), name)
                dirPath.mkdirs()
                println(s"Writing to ${dirPath.getPath}")
                csvwrite(new File(dirPath.getPath, "block_features.csv"), extract._1.blockFeatures)
                csvwrite(new File(dirPath.getPath, "edge_features.csv"), extract._1.edgeFeatures)
            } else {
                println(s"Skipping ${pair._1}")
            }
        }

        println("Done!\n")
    }
}