package com.nexa.ai.tokenizer;

import java.util.*;

public class Vocabulary {

    public static final String UNK_TOKEN = "<UNK>";
    public static final String PAD_TOKEN = "<PAD>";

    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;

    public Vocabulary() {
        tokenToId = new LinkedHashMap<>();
        idToToken = new LinkedHashMap<>();

        addToken(UNK_TOKEN);
        addToken(PAD_TOKEN);
    }

    public int addToken(String token) {
        if (token == null || token.isBlank()) {
            throw new IllegalArgumentException("Token cannot be null or blank");
        }

        if (tokenToId.containsKey(token)) {
            return tokenToId.get(token);
        }

        int id = tokenToId.size();

        tokenToId.put(token, id);
        idToToken.put(id, token);

        return id;
    }

    public int getId(String token) {
        if (token == null) {
            return tokenToId.get(UNK_TOKEN);
        }

        return tokenToId.getOrDefault(
                token,
                tokenToId.get(UNK_TOKEN)
        );
    }

    public String getToken(int id) {
        String token = idToToken.get(id);

        if (token == null) {
            throw new IllegalArgumentException(
                    "Unknown token ID: " + id
            );
        }

        return token;
    }

    public boolean contains(String token) {
        return tokenToId.containsKey(token);
    }

    public int size() {
        return tokenToId.size();
    }

    public Map<String, Integer> getTokenToId() {
        return Collections.unmodifiableMap(tokenToId);
    }

    public Map<Integer, String> getIdToToken() {
        return Collections.unmodifiableMap(idToToken);
    }
}