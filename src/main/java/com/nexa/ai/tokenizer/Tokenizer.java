package com.nexa.ai.tokenizer;

import java.util.List;

public interface Tokenizer {

    List<String> tokenize(String text);

    List<Integer> encode(String text);

    String decode(List<Integer> tokenIds);
}