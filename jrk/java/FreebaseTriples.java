package jrk.java;
import java.io.*;
import java.util.*;
import gnu.trove.map.hash.*;
import gnu.trove.map.*;
import gnu.trove.set.hash.*;
import gnu.trove.set.*;
import gnu.trove.iterator.*;

public class FreebaseTriples {
    static Set<String> entSet = null;
    static final String prefix = "http://rdf.freebase.com/ns/";

    public static String clean(String str) {
        if (str.startsWith("<"))
            str = str.substring(1, str.length() - 1);
        if (str.startsWith(prefix))
            return str.substring(prefix.length());
        else
            return null;
    }


	public static void loadEntSet(String path) {
	    entSet = new HashSet<>();

        try {
            System.out.println("loading ent from " + path);
            BufferedReader br = new BufferedReader(new FileReader(path));
            int count = 0;
            for (String line; (line = br.readLine()) != null; ) {
                String[] strs = line.split("\t");
                if (strs.length != 2){
                    System.out.println(line);
                    continue;
                }

                entSet.add(strs[0]);
                count++;
                if (count % (int)1e6 == 0) {
                    System.out.print(Integer.toString(count) + "\r");
                    //break;
                }
            }

            System.out.format("total %d freebase entities\n", entSet.size());
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

	public static void getTriples(String fbPath, String outPath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(fbPath));
            BufferedWriter bw = new BufferedWriter(new FileWriter(outPath));
            int count = 0;
            int nTriples = 0;

            for (String line; (line = br.readLine()) != null; ) {
                String[] strs = line.split("\t");
                count++;
                if (count % (int)1e6 == 0) {
                    System.out.print(String.format("%15d\t%15d", count, nTriples) + "\r");
                    //break;
                }
 
                String hstr = clean(strs[0]);
                String tstr = clean(strs[2]);
                String rstr = clean(strs[1]);
                if (hstr == null || tstr == null || rstr == null)
                    continue;
                if (!entSet.contains(hstr) || !entSet.contains(tstr))
                    continue;
                bw.write(hstr + "\t" + rstr + "\t" + tstr + "\n");
                nTriples++;

           }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

	public static void main(String[] args) {
        {
            String entPath = "data/freebase/freebase-entity.txt";
	        System.out.println("loading entities from " + entPath);
            loadEntSet(entPath);

            String fbPath = "../freebase2tacred/data/freebase-rdf-latest";
            String outPath = "data/freebase/freebase-triples.txt";
            System.out.println("get triples from freebase " + fbPath);
            getTriples(fbPath, outPath);
        }
    }
}
