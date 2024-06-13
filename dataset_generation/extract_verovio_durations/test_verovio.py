import verovio
from lxml import etree


"""
<mei xmlns="http://www.music-encoding.org/ns/mei">
  <meiHead>
    <fileDesc>
      <titleStmt>
        <title>Single Note Extraction</title>
      </titleStmt>
    </fileDesc>
  </meiHead>
  <music>
    <body>
      <mdiv>
        <score>
          <scoreDef>
            <clef shape="G" line="2"/>
            <keySig sig="0"/>
            <mensur sign="C" num="4"/>
          </scoreDef>
          <section>
            <measure>
              <staff>
                <layer n="14">
                  <note xml:id="note-0000000148993359" dur="16" oct="5" pname="a"/>
                </layer>
              </staff>
            </measure>
          </section>
        </score>
      </mdiv>
    </body>
  </music>
</mei>
"""

string="""
<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei">
  <meiHead>
    <fileDesc>
      <titleStmt>
        <title>Test Note</title>
      </titleStmt>
    </fileDesc>
  </meiHead>
  <music>
    <body>
      <mdiv>
        <score>
          <scoreDef>
            <staffGrp>
              <staffDef n="1" lines="5" />
            </staffGrp>
          </scoreDef>
          <section>
            <measure n="1">
              <staff n="1">
                <layer n="1">
                  <note xml:id="n1" dur="4" pname="c" oct="4"/>
                </layer>
              </staff>
            </measure>
          </section>
        </score>
      </mdiv>
    </body>
  </music>
</mei>
"""
vrvToolkit = verovio.toolkit()
vrvToolkit.loadData(string)