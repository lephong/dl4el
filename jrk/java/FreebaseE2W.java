package jrk.java;
import java.io.*;
import java.util.*;
import gnu.trove.map.hash.*;
import gnu.trove.map.*;
import gnu.trove.set.hash.*;
import gnu.trove.set.*;
import gnu.trove.iterator.*;

public class FreebaseE2W {
    static int nHop = 0;
    static StrIntMap dict = null;
    static StrIntMap entId = null;
    static TIntObjectHashMap<TIntHashSet> curE2WId = null;
    static TIntObjectHashMap<TIntHashSet> preE2WId = null;
    static final String prefix = "http://rdf.freebase.com/ns/";

    public static String clean(String str) {
        if (str.startsWith("<"))
            str = str.substring(1, str.length() - 1);
        if (str.startsWith(prefix))
            return str.substring(prefix.length());
        else
            return null;
    }

    public static TIntHashSet getWords(String str, TIntHashSet ret) {
        String[] words = str.split("[_ ./]");
        if (ret == null)
            ret = new TIntHashSet();
        if (words.length < 20) {
            for (String w: words) {
                int id = dict.add(w);
                ret.add(id);
            }
        }
        return ret;
    }

    public static void loadWordId(String dir) {
        dict = new StrIntMap();

        try {
            System.out.println("loading words from " + dir + "/wordId.ser");
            File f = new File(dir + "/wordId.ser");
            if (f.exists()) {
                ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
                dict = (StrIntMap) ois.readObject();
            }
			System.out.format("total %d words\n", dict.size());
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }


	public static void loadPreHopE2W(String dir, String fname) {
        preE2WId = new TIntObjectHashMap<>();
	    entId = new StrIntMap();
        dict = new StrIntMap();

        try {
            System.out.println("loading words from " + dir + "/wordId.ser");
            File f = new File(dir + "/wordId.ser");
            if (f.exists()) {
                ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
                dict = (StrIntMap) ois.readObject();
            }
			System.out.format("total %d words\n", dict.size());

            System.out.println("loading prehopE2W from " + dir + "/" + fname);
            BufferedReader br = new BufferedReader(new FileReader(dir + "/" + fname));
            int count = 0;
            for (String line; (line = br.readLine()) != null; ) {
                String[] strs = line.split("\t");
                Integer eId = entId.add(strs[0]);
                preE2WId.put(eId, getWords(strs[1], null));
                count++;
                if (count % (int)1e6 == 0) {
                    System.out.print(Integer.toString(count) + "\r");
                    //break;
                }
            }

            System.out.format("total %d freebase entities\n", preE2WId.size());
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

	public static void nextHop(String path) {
        try {
            curE2WId = new TIntObjectHashMap<>();
            BufferedReader br = new BufferedReader(new FileReader(path));
            StrIntMap relId = new StrIntMap();
            TIntObjectHashMap<TIntHashSet> r2wId = new TIntObjectHashMap<>();
            int count = 0;

            for (String line; (line = br.readLine()) != null; ) {
                String[] strs = line.split("\t");

                String hstr = clean(strs[0]);
                String tstr = clean(strs[2]);
                String rstr = clean(strs[1]);
                if (hstr == null || tstr == null || rstr == null)
                    continue;

                int h = entId.str2int.get(hstr);
                int t = entId.str2int.get(tstr);
                if (h == entId.str2int.getNoEntryValue() || t == entId.str2int.getNoEntryValue())
                    continue;

                TIntHashSet hwords = preE2WId.get(h);
                TIntHashSet twords = preE2WId.get(t);

                int r = relId.str2int.get(rstr);
                TIntHashSet rwords = null;
                if (r == entId.str2int.getNoEntryValue()) {
                    r = relId.add(rstr);
                    rwords = getWords(rstr, null);
                    r2wId.put(r, rwords);
                }
                else 
                    rwords = r2wId.get(r);
                boolean isTypeInstance = rstr.endsWith("type.type.instance");
                boolean isObjType = rstr.endsWith("type.object.type") || rstr.endsWith("prominent_type");

                // for h
                {
                    TIntHashSet words = curE2WId.get(h);
                    if (words == null) {
                        words = new TIntHashSet();
                        curE2WId.put(h, words);
                    }
                    if (!isTypeInstance) {
                        words.addAll(twords);
                        words.addAll(rwords);
                    }
                }
 
                // for t
                {
                    TIntHashSet words = curE2WId.get(t);
                    if (words == null) {
                        words = new TIntHashSet();
                        curE2WId.put(t, words);
                    }
                    if (!isObjType) {
                        words.addAll(hwords);
                    }
                    if (!(isObjType || isTypeInstance))
                        words.addAll(rwords);
                }

                count++;
                if (count % (int)1e6 == 0) {
                    System.out.print(Integer.toString(count) + "\r");
                    //break;
                }
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static void printRelWords(String path) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(path));
            int count = 0;
            Set<String> allWords = new HashSet<>();

            for (String line; (line = br.readLine()) != null; ) {
                String[] strs = line.split("\t");
                String rstr = clean(strs[1]);
                if (rstr != null) {
                    String[] words = rstr.split("[_ ./]");
                    for (String w: words)
                        allWords.add(w);
                }

                count++;
                if (count % 1000000 == 0)
                    System.out.print(count + "\r");
            }

            BufferedWriter bw = new BufferedWriter(new FileWriter("relWords.txt"));
            for (String w: allWords)
                bw.write(w + "\n");
            bw.close();
       }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static void printEntType(String path) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(path));
            BufferedWriter bw = new BufferedWriter(new FileWriter("freebase-ent-type.txt"));
            Map<String, List<String>> e2cList = new HashMap<>();
            int count = 0;

            for (String line; (line = br.readLine()) != null; ) {
                String[] strs = line.split("\t");
                if (strs[1].endsWith("type.instance>")) {
                    count++;
                    if (count % 1000 == 0) {
                        System.out.print(count + "\r");
                        //break;
                    }

                    String hstr = clean(strs[0]);
                    String tstr = clean(strs[2]);
                    List<String> cList = e2cList.get(tstr);
                    if (cList == null) {
                        cList = new ArrayList<>();
                        e2cList.put(tstr, cList);
                    }
                    cList.add(hstr);
                }
            }

            for (Map.Entry<String, List<String>> item: e2cList.entrySet()) {
                bw.write(item.getKey() + "\t");
                for (String cat: item.getValue())
                    bw.write(cat + " ");
                bw.write("\n");
            }
            bw.close();
       }
        catch(Exception e) {
            e.printStackTrace();
        }
    }



    public static void combineFilesNextHop(String dir, int n) {
        try {
            curE2WId = new TIntObjectHashMap<>();
            for (int i = 0; i < n ; i++) {
                String ppath = dir + String.format("/e2w_%02d.ser", i);
                System.out.println("merging with " + ppath);
                
                File f = new File(ppath);
                if (!f.exists()) {
                    System.out.println("STOP");
                    break;
                }
                
                ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
                TIntObjectHashMap curE2WId_i = (TIntObjectHashMap) ois.readObject();
                for (TIntObjectIterator<TIntHashSet> iter = curE2WId_i.iterator(); iter.hasNext(); ) {
                    iter.advance();
                    int eId = iter.key();
                    TIntHashSet ws_i = iter.value();
                    TIntHashSet ws = curE2WId.get(eId);
                    if (ws == null)
                        curE2WId.put(eId, ws_i);
                    else
                        ws.addAll(ws_i);
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void saveWordIdTxt(String dir) {
        try {
            System.out.println("write wordId");
            BufferedWriter bw = new BufferedWriter(new FileWriter(dir + "/wordId.txt"));
            for (TObjectIntIterator<String> iter = dict.str2int.iterator(); iter.hasNext(); ) {
                iter.advance();
                String word = iter.key();
                bw.write(word + "\n");
            }
            bw.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void saveToFile(String dir, String fname, String format) {
        try {
            // write dict using ser
            System.out.println("save wordId to " + dir + "/wordId.ser");
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(dir + "/wordId.ser"));
            oos.writeObject(dict);

            if (format.equals("txt")) {
                System.out.println("write e2w");
                BufferedWriter bw = new BufferedWriter(new FileWriter(dir + "/" + fname));
                for (TIntObjectIterator<TIntHashSet> iter = curE2WId.iterator(); iter.hasNext(); ) {
                    iter.advance();
                    int eId = iter.key();
                    TIntHashSet ws = iter.value();
                    bw.write(entId.int2str.get(eId) + "\t");
                    for (TIntIterator wIter = ws.iterator(); wIter.hasNext(); ) {
                        int wId = wIter.next();
                        bw.write(dict.int2str.get(wId) + " ");
                    }
                    bw.write("\n");
                }
                bw.close();
            }

            else if (format.equals("ser")) {
                System.out.println("save e2w to " + dir + "/" + fname);
                oos = new ObjectOutputStream(new FileOutputStream(dir + "/" + fname));
                oos.writeObject(curE2WId);
            }

        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

	public static void main(String[] args) {
        String mode = args[0];

        if (mode.equals("process")) {
            int portionId = Integer.parseInt(args[1]);

            String dir = "data/freebase/";
		    String firstHopE2W_fname = "freebase-entity.txt";

	        System.out.println("loading entities from " + dir + firstHopE2W_fname);
            loadPreHopE2W(dir, firstHopE2W_fname);

            String fbPath = dir + String.format("freebase-rdf-%02d", portionId);
            System.out.println("get words from freebase " + fbPath);
            nextHop(fbPath);
        
            String format = "ser";
            String fname = String.format("e2w_%02d.%s", portionId, format);
            saveToFile(dir, fname, format);
        }

        else if (mode.equals("combine")) {
            int nPortion = Integer.parseInt(args[1]);
            String dir = "data/freebase/";
		    String firstHopE2W_fname = "freebase-entity.txt";

	        System.out.println("loading entities from " + dir + firstHopE2W_fname);
            loadPreHopE2W(dir, firstHopE2W_fname);

            String fname = "e2w.txt";
            combineFilesNextHop(dir, nPortion);
            saveToFile(dir, fname, "txt");
        }

        else if (mode.equals("print_wordId")) {
            String dir = "data/freebase/";
            System.out.println("load wordId");
            loadWordId(dir);
            System.out.println("save to file");
            saveWordIdTxt(dir);
        }

        else if (mode.equals("print_relWords")) {
            String fbPath = "../freebase2tacred/data/freebase-rdf-latest";
            System.out.println("process" + fbPath);
            printRelWords(fbPath);
        }
        else if (mode.equals("print_ent_type")) {
            String fbPath = "../freebase2tacred/data/freebase-rdf-latest";
            System.out.println("process" + fbPath);
            printEntType(fbPath);
        }
    }
}
