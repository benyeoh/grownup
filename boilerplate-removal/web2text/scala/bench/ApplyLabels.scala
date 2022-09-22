package bench

import java.io.File
import scala.io.Source
import breeze.linalg.csvwrite
import ch.ethz.dalab.web2text.cdom.CDOM
import ch.ethz.dalab.web2text.utilities.Util
import ch.ethz.dalab.web2text.output.CleanTextOutput


object ApplyLabels {

    def main(args: Array[String]): Unit = {
        if (args.length < 3) {
            throw new IllegalArgumentException("Expecting arguments: (1) input html dir, (2) inpout labels dir, (3) output dir")
        }

        applyLabelsToPage(args(0), args(1), args(2))
    }

    def findHTMLLabelPair(htmlDir: String, labelDir: String) : List[(String, String)] = {
        def getListOfFiles(dir: String, ext: String): List[String] = {
            val file = new File(dir)
            file.listFiles.filter(_.isFile)
                .filter(_.getName.endsWith(ext))
                .map(_.getPath).toList
        }

        val allHtml = getListOfFiles(htmlDir, ".html")
        val allLabels = getListOfFiles(labelDir, ".csv")
        
        val labelsMap = (allLabels.map{path => path.split('/').last.split(".csv")(0)}
                         zip
                         allLabels).toMap

        val filteredPairs = allHtml.filter{path => labelsMap.contains(path.split('/').last.split(".html")(0))}
                           .map(path => (path, labelsMap(path.split('/').last.split(".html")(0))))

        return filteredPairs
    }

    def applyLabelsToPage(htmlDir: String, labelDir: String, outputDir: String) = {
        val dirPath = new File(outputDir)
        dirPath.mkdirs()

        val filteredPairs = findHTMLLabelPair(htmlDir, labelDir)
        filteredPairs.foreach( pairs => {
            val outFileName = pairs._1.split('/').last.split(".html")(0)
            val outFilePath = new File(outputDir, s"${outFileName}.txt").getPath
            println(s"Writing results to: ${outFilePath}")
            val p = Page(pairs._1, null)
            val cdom = CDOM(p.source mkString)
            val labels = Util.loadFile(pairs._2).split(",").map(_.toInt)
            val extractedText = CleanTextOutput(cdom, labels)
            Util.save(outFilePath, extractedText)
        })
    }
}
