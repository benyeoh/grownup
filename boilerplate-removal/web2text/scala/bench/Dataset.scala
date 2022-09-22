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
import java.io.File
import java.util.NoSuchElementException
import ch.ethz.dalab.web2text.output.CsvDatasetWriter
import ch.ethz.dalab.web2text.utilities.Warc
import ch.ethz.dalab.web2text.output.CleanTextOutput
import scala.util.{Try,Success,Failure}


object Dataset {
    val featureExtractor = FeatureExtractor(
            DuplicateCountsExtractor
            + LeafBlockExtractor
            + AncestorExtractor(NodeBlockExtractor + TagExtractor(mode="node"), 1)
            + AncestorExtractor(NodeBlockExtractor, 2)
            + RootExtractor(NodeBlockExtractor)
            + TagExtractor(mode="leaf"),
            TreeDistanceExtractor + BlockBreakExtractor + CommonAncestorExtractor(NodeBlockExtractor)
        )

    def parseCDOM(source: String) : CDOM = {
        try {
            return CDOM(source)
        } catch {
            case e: java.util.NoSuchElementException => { 
                println(s"Error parsing ${source}")
            }
        }

        return null
    }

    def evalAlign(page: Page) : String = {
        try {
            return page.aligned
        } catch {
            case _: Throwable => {
                println("Problem evaluation page alignment!")
            }
        }

        return null
    }

    def findHTMLCleanPair(htmlDir: String, cleanDir: String) : List[(String, String)] = {
        def getListOfFiles(dir: String, ext: String): List[String] = {
            val file = new File(dir)
            file.listFiles.filter(_.isFile)
                .filter(_.getName.endsWith(ext))
                .map(_.getPath).toList
        }

        val allHtml = getListOfFiles(htmlDir, ".html")
        val allClean = getListOfFiles(cleanDir, ".txt")
        
        val cleanMap = (allClean.map{path => path.split('/').last.split('.')(0).split("-cleaned")(0)}
                        zip
                        allClean).toMap

        val filteredPairs = allHtml.filter{path => cleanMap.contains(path.split('/').last.split(".html")(0))}
                           .map(path => (path, cleanMap(path.split('/').last.split(".html")(0))))

        return filteredPairs
    }

    def extractAll(filePairs: List[(String, String)]): Stream[(PageFeatures, Vector[Int], Vector[String])] = {
        for {
            ((htmlPath, cleanPath), i) <- filePairs.zipWithIndex.toStream
            _ = println(s"${i + 1} - Extracting feats/labels/texts: ${htmlPath}, ${cleanPath}")
            p = Page(htmlPath, cleanPath)
            cdom = parseCDOM(p.source)
            align = evalAlign(p)
            feats = if (cdom != null && align != null) featureExtractor(cdom) else null
            labels = if (cdom != null && align != null) Alignment.extractLabels(cdom, align) else null
            texts = if (cdom != null && align != null) cdom.leaves map {n => n.text} else null
        } yield {
            if (feats != null) {
                assert(feats.nBlocks == labels.length)
                assert(texts.length == labels.length)
            }
            (feats, labels, texts)
        }
    }

   def extractFeatsAndText(files: List[String]): List[(PageFeatures, Vector[String])] = {
        for {
            (htmlPath, i) <- files.zipWithIndex
            _ = println(s"${i + 1} - Extracting feats/text: ${htmlPath}")
            p = Page(htmlPath, null)
            cdom = parseCDOM(p.source)
            feats = if (cdom != null) featureExtractor(cdom) else null
            texts = if (cdom != null) cdom.leaves map {n => n.text} else null
        } yield {
            if (feats != null) {            
                assert(feats.nBlocks == texts.length)
            }
            (feats, texts)
        }
    }
}