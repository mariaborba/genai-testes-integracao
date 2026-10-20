 eid in criados:
                resp = cliente_app.get(f"/episodios/{eid}")
                assert resp.status_code == 404
        finally:
            cliente_app.post = original_post
```