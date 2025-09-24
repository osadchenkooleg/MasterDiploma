func wrpath(ctxt *Link, b *Biobuf, p string) {
	var i int
	var n int
	if ctxt.Windows == 0 || !strings.Contains(p, `\`) {
		wrstring(b, p)
		return
	}
	n = len(p)
	wrint(b, int64(n))
	for i = 0; i < n; i++ {
		var tmp int
		if p[i] == '\\' {
			tmp = '/'
		} else {
			tmp = int(p[i])
		}
		Bputc(b, tmp)
	}
}