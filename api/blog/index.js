export default async function handler(req, res) {
  const backendUrl = 'https://bgremover-backend-121350814881.us-central1.run.app/blog';
  
  try {
    const response = await fetch(backendUrl, {
      method: req.method,
      headers: {
        'Content-Type': 'text/html',
      },
    });
    
    if (!response.ok) {
      return res.status(response.status).json({ error: 'Backend request failed' });
    }
    
    const html = await response.text();
    res.setHeader('Content-Type', 'text/html');
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
    return res.status(200).send(html);
  } catch (error) {
    console.error('Error proxying blog request:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

