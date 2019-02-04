package jrk.java;
import java.util.*;
import java.io.Serializable;
import gnu.trove.map.hash.*;


public class StrIntMap implements Serializable {
    public TObjectIntHashMap<String> str2int;
    public ArrayList<String> int2str;

	StrIntMap() {
        str2int = new TObjectIntHashMap<>();
        int2str = new ArrayList<>();
    }

    StrIntMap(int capacity) {
        str2int = new TObjectIntHashMap<>(capacity);
        int2str = new ArrayList<>(capacity);
    }

    int add(String str) {
        int id = str2int.get(str);
        if (id == str2int.getNoEntryValue()) {
            int _id = int2str.size();
            str2int.put(str, _id);
            int2str.add(str);
            return _id;
        }
        else {
            return id;
        }
    }

    void add(String str, int id) {
        int _id = add(str);
        if (_id != id)
            throw new IllegalArgumentException("words must be sorted by their ids");
    }

    void clear() {
        str2int.clear();
        int2str.clear();
    }

	int size() {
		return int2str.size();
	}
}



