package com.nexa.ai.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SimpleTokenizer implements Tokenizer {

    private static final Pattern TOKEN_PATTERN =
            Pattern.compile(
                    "[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_\\s]"
            );

    private final Vocabulary vocabulary;

    public SimpleTokenizer(Vocabulary vocabulary) {
        if (vocabulary == null) {
            throw new IllegalArgumentException("Vocabulary cannot be null");
        }

        this.vocabulary = vocabulary;
    }

    @Override
    public List<String> tokenize(String text) {

        if (text == null || text.isBlank()) {
            return List.of();
        }

        List<String> tokens = new ArrayList<>();

        Matcher matcher = TOKEN_PATTERN.matcher(text);

        while (matcher.find()) {
            tokens.add(matcher.group());
        }

        return tokens;
    }

    @Override
    public List<Integer> encode(String text) {

        List<String> tokens = tokenize(text);

        List<Integer> ids = new ArrayList<>();

        for (String token : tokens) {
            ids.add(vocabulary.getId(token));
        }

        return ids;
    }

    @Override
    public String decode(List<Integer> tokenIds) {

        if (tokenIds == null) {
            throw new IllegalArgumentException(
                    "Token IDs cannot be null"
            );
        }

        List<String> tokens = new ArrayList<>();

        for (Integer id : tokenIds) {

            if (id == null) {
                throw new IllegalArgumentException(
                        "Token ID cannot be null"
                );
            }

            tokens.add(vocabulary.getToken(id));
        }

        return String.join(" ", tokens);
    }

    public Vocabulary getVocabulary() {
        return vocabulary;
    }
}