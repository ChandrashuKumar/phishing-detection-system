import {NextResponse} from "next/server";

export async function POST(req){
    const {url} = await req.json();

    if(!url){
        return NextResponse.json({error: "Missing URL"}, {status: 400});
    }

    try {
        const res = await fetch(`${process.env.BACKEND_API_URL}/api/url/detect`,{
            method: "POST",
            headers:{
                "Content-Type": "application/json"
            },
            body: JSON.stringify({url})
        });

        const data = await res.json();

        if(!res.ok){
            return NextResponse.json({error: data.error || "Backend error"}, {status: res.status});
        }

        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({error: error.message}, {status: 500});
    }

}