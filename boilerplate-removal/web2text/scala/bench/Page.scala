package bench

import java.io.File
import scala.io.{Source,Codec}
import org.jsoup.Jsoup

import ch.ethz.dalab.web2text.utilities.Util
import ch.ethz.dalab.web2text.utilities.Util.codec
import ch.ethz.dalab.web2text.alignment.Alignment
import ch.ethz.dalab.web2text.features.{FeatureExtractor,PageFeatures}
import ch.ethz.dalab.web2text.cdom.CDOM
import ch.ethz.dalab.web2text.classification.PerformanceStatistics


case class Page(htmlPath: String, cleanPath: String) {

    private val UrlTitleEncoding = """(?i)<text id="(.*?)" title="(.*?)" encoding="(.*?)">""".r
    private val UrlEncoding = """(?i)<text id="(.*?)" encoding="(.*?)">""".r

    def charsetMap(given: String) = given.toLowerCase match {
        case "windows-1252" => "windows-1252"
        case "iso-8859-1" => "iso-8859-1"
        case "utf-8" => "utf-8"
        case "utf8" => "utf-8"
        case _ => ""
    }

    lazy val source: String = {
        val source = Source.fromFile(htmlPath)
        val lines = source.getLines.toVector

        val (url: String, title: String, encoding: String) = lines(0) match {
            case UrlTitleEncoding(url, title, encoding) => (url, title, charsetMap(encoding))
            case UrlEncoding(url, encoding) => (url, "", charsetMap(encoding))
            case _ => ("","","")
        }

        if (url != "" || title != "" || encoding != "")
            lines.slice(1, lines.length-1) mkString "\n"
        else
            lines mkString "\n"
    }

    lazy val clean: String = {
        if (cleanPath != null) {
            val source = Source.fromFile(cleanPath)
            val text = source.getLines mkString "\n"

            val contents = if (text.startsWith("URL:")) text.lines.drop(1).mkString("\n") else text

            contents.replaceAll(
                """(?i)(?m)^\s*<l>\s*(»|\*|\d{1,2}\.\s)\s*|<(l|h|p)>\s*|^\s*(_{10,}|-{10,})\s*$|^\s*""","")
                .trim
        } else {
            ""
        }
    }

    lazy val aligned: String = Alignment.alignment(source, clean)
}
