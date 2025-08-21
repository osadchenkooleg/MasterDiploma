public class UniquePlagiarismProbe {
    public static String fingerprint(int n) {
        String s = "ua_kpi_masters_" + n + "_x9Z!"; // унікальний текст
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            b.append((char)(s.charAt(i) ^ (i * 17 + 11)));
        }
        return b.reverse().toString();
    }
}